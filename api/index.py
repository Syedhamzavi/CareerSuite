import base64
import html as html_lib
import ipaddress
import json
import os
import re
import smtplib
import socket
from email.message import EmailMessage
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------- load .env FIRST, before any setting is read ----------
BASE_DIR = Path(__file__).parent
for _p in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True, encoding="utf-8-sig")
        print(f".env loaded from: {_p}")
        break
else:
    print("WARNING: no .env file found")

print("GEMINI key present:", bool(os.getenv("GEMINI_API_KEY", "").strip()))

# ---------- config ----------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "").strip().strip("'\"")
# Google shows App Passwords in groups of four with spaces; remove them and any quotes
SMTP_PASSWORD = re.sub(r"\s+", "", os.getenv("SMTP_APP_PASSWORD", "")).strip("'\"")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Syed Mujataba Hamzavi").strip()

MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024
EMAIL_RE = re.compile(r"^[^@\s,;]+@[^@\s,;]+\.[^@\s,;]+$")

PROFILE_PATH = BASE_DIR / "master_profile.json"
LINKS_PATH = BASE_DIR / "links.json"
PUBLIC_DIR = BASE_DIR.parent / "public"

AUTH_HELP = (
    "Gmail rejected the login. Use a 16-character App Password "
    "(Google Account > Security > 2-Step Verification > App passwords) in SMTP_APP_PASSWORD, "
    "and your full Gmail address in SMTP_USER."
)
NET_HELP = (
    "Could not reach the mail server on ports 465 or 587. A firewall, antivirus or your network "
    "(college and office Wi-Fi often block email) is in the way. Try a mobile hotspot, or use Open in Gmail."
)

app = FastAPI(title="Syed AI Resume & Career Suite API")

if PUBLIC_DIR.exists():
    app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")


@app.get("/")
def root_redirect():
    return RedirectResponse(url="/public/index.html")


# ---------- schemas ----------
class GenerateRequest(BaseModel):
    jd_text: str
    extra_notes: str = ""
    linkedin_url: str = "https://www.linkedin.com/in/syed-hamzavi12"
    location: str = "Bengaluru"


class SkillCategory(BaseModel):
    category_name: str = Field(description="Category title e.g. Data Analysis, BI & Visualization, Technical Support")
    skills_list: str = Field(description="Comma-separated list of 5 to 9 relevant skills matching the JD")


class ExperienceItem(BaseModel):
    role: str
    company: str
    duration: str
    bullets: list[str] = Field(description="3 to 5 high-impact bullets tailored to the JD")


class ProjectItem(BaseModel):
    title: str = Field(description="Project Name | Tech Stack")
    bullets: list[str] = Field(description="2 to 3 concise metric-driven bullets tailored to the JD")


class CertificationItem(BaseModel):
    title: str
    year: str


class TailoredResumeResponse(BaseModel):
    headline: str = Field(description="2 or 3 pipe-separated target job titles matching the JD")
    summary: str = Field(description="3 to 4 sentence tailored summary matching Syed's background")
    skills: list[SkillCategory] = Field(description="5 to 6 categorized skill rows")
    experience: list[ExperienceItem] = Field(description="1 or 2 relevant internship experiences")
    projects: list[ProjectItem] = Field(description="2 or 3 most relevant projects")
    certifications: list[CertificationItem] = Field(description="Top 5 certifications")
    ats_match_score: int = Field(description="ATS keyword match percentage (0-100)")
    matched_keywords: list[str] = Field(description="Top 8-12 matched keywords from JD")
    cover_letter: str = Field(description="A professional, compelling 3-paragraph cover letter tailored to the target JD and candidate's real profile.")
    cold_email_subject: str = Field(description="Subject line for the recruiter cold email, under 70 characters, naming the role. No placeholders.")
    cold_email: str = Field(description="Concise cold email body for a recruiter or hiring manager about this specific role. No subject line. No bracketed placeholders.")
    suggestions: list[str] = Field(description="4 to 6 specific, actionable improvements for this application, based on gaps between the JD and the real profile.")
    skills_to_build: list[str] = Field(description="Up to 5 skills or tools the JD wants that are NOT in the profile and are worth learning. Empty list if none.")
    interview_points: list[str] = Field(description="4 to 5 topics or questions this JD makes likely in an interview, tied to the candidate's real projects.")


class SendEmailRequest(BaseModel):
    to_email: str
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=8000)
    attachment_b64: Optional[str] = None
    attachment_name: str = "Resume.pdf"


class FetchJDRequest(BaseModel):
    url: str


# ---------- project and certificate links ----------
def _valid_url(u) -> bool:
    return isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def _match(title: str, key: str) -> bool:
    t, k = title.lower().strip(), key.lower().strip()
    if not t or not k:
        return False
    if k in t or t in k:
        return True
    a, b = _tokens(t), _tokens(k)
    common = len(a & b)
    return bool(a and b) and common >= 2 and common >= 0.6 * min(len(a), len(b))


def _title_of(d: dict) -> str:
    for k in ("title", "name", "project", "project_name", "certification", "course"):
        if isinstance(d.get(k), str):
            return d[k]
    return ""


def _urls_of(d: dict) -> list:
    pairs, used = [], set()
    for k, v in d.items():
        if not _valid_url(v):
            continue
        key = k.lower()
        if "github" in key or "repo" in key or "code" in key:
            label = "GitHub"
        elif any(x in key for x in ("live", "demo", "dashboard", "tableau", "power", "app", "site", "deploy")):
            label = "Live"
        else:
            label = "Link"
        if label not in used:
            used.add(label)
            pairs.append((label, v.strip()))
    return pairs


def _collect_link_entries(profile_data: dict):
    """links.json wins; master_profile.json is the fallback. First match wins."""
    project_entries, cert_entries = [], []

    if LINKS_PATH.exists():
        try:
            raw = json.loads(LINKS_PATH.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print("links.json is not valid JSON:", e)
            raw = {}
        for key, val in (raw.get("projects") or {}).items():
            if isinstance(val, str):
                val = {"Link": val}
            pairs = [(lbl, u.strip()) for lbl, u in (val or {}).items() if _valid_url(u)]
            if pairs:
                project_entries.append((key, pairs))
        for key, val in (raw.get("certifications") or {}).items():
            if _valid_url(val):
                cert_entries.append((key, val.strip()))

    for top_key, items in profile_data.items():
        if not isinstance(items, list):
            continue
        name = top_key.lower()
        for item in items:
            if not isinstance(item, dict):
                continue
            title = _title_of(item)
            if not title:
                continue
            if "project" in name:
                pairs = _urls_of(item)
                if pairs:
                    project_entries.append((title, pairs))
            elif "cert" in name:
                pairs = _urls_of(item)
                if pairs:
                    cert_entries.append((title, pairs[0][1]))

    return project_entries, cert_entries


def _attach_links(tailored: dict, profile_data: dict) -> dict:
    project_entries, cert_entries = _collect_link_entries(profile_data)
    found = {"projects": 0, "certifications": 0}

    for p in tailored.get("projects", []):
        for key, pairs in project_entries:
            if _match(p.get("title", ""), key):
                p["links"] = [{"label": lbl, "url": u} for lbl, u in pairs]
                found["projects"] += 1
                break

    for c in tailored.get("certifications", []):
        for key, url in cert_entries:
            if _match(c.get("title", ""), key):
                c["url"] = url
                found["certifications"] += 1
                break

    print(f"links attached: {found['projects']} projects, {found['certifications']} certifications")
    return found


# ---------- routes ----------
@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "gemini_key_present": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "email_enabled": bool(SMTP_USER and SMTP_PASSWORD),
    }


@app.get("/api/email-status")
def email_status():
    return {"enabled": bool(SMTP_USER and SMTP_PASSWORD), "from": SMTP_USER}


@app.post("/api/generate")
def generate_resume(req: GenerateRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set. Add it to your .env file.")

    if not PROFILE_PATH.exists():
        raise HTTPException(status_code=500, detail=f"master_profile.json not found at {PROFILE_PATH}")

    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        profile_data = json.load(f)

    client = genai.Client(api_key=api_key)

    system_prompt = """
    You are an elite, world-class Executive Career Coach and Senior ATS Resume Optimization Expert.
    Your mission is to meticulously tailor Syed Mujataba Hamzavi's master profile to perfectly align with the target Job Description (JD).

    You must output valid JSON strictly adhering to the requested TailoredResumeResponse schema.

    CORE EXECUTION DIRECTIVES:
    1. ABSOLUTE TRUTHFULNESS: Use only Syed's authentic experience (AiVariant Data Analyst Internship, Eunoia Labs Backend Internship), real metrics, B.E. in AI & Data Science from CIT VTU (June 2025, CGPA 7.85/10), publications (IJPREMS Nov 2024), Shark Tank participation (Dec 2024), and verified certifications (NASSCOM, ExcelR, Forage TATA, Internshala, Great Learning). Never hallucinate companies, degrees, or unverified facts.
    2. SMART ROLE-BASED ADAPTATION:
       - IF DATA ANALYST / BI ROLE: Highlight his SQL stars-schema joins, Adventure Works Tableau/Power BI metrics (60.4K+ orders, ₹29.36M+ sales), DAX, Excel, and analytics workflow.
       - IF TECHNICAL / IT / APPLICATION SUPPORT ROLE: Emphasize troubleshooting, data validation, query resolution, issue diagnosis, stakeholder management, and technical documentation.
       - IF AI / ML / PYTHON / SOFTWARE ENGINEERING ROLE: Showcase his Python backend internships, SPEDOCITY app (40% delivery efficiency boost), Finvisor Streamlit app (85% ML accuracy), Multiple Disease Prediction (90%+ accuracy), and cryptography password manager.
    3. ATS KEYWORD OPTIMIZATION & FORMATTING:
       - Extract core technical terms, tools, and action verbs from the JD and weave them naturally into his summary, skill rows, and bullet points.
       - Ensure the total volume of text across Experience and Projects is concise enough to fit strictly within a single A4 printable page budget.
    4. CAREER SUITE GENERATION:
       - Draft a professional, compelling 3-paragraph Cover Letter explicitly referencing the target company/role and mapping Syed's core technical strengths to their business needs.
       - Draft a sharp, high-converting Cold Email for outreach to recruiters or hiring managers. The email will be SENT AS WRITTEN, so: open with "Hello," (no recruiter name), never use bracketed placeholders like [Company] (use the company name from the JD if it exists, otherwise just refer to the role), keep it under 130 words, do not include a subject line in the body, and sign off with "Best regards," followed by "Syed Mujataba Hamzavi".
       - Write a separate cold_email_subject of under 70 characters that names the role.
    5. SUGGESTIONS:
       - suggestions: specific and actionable (what to add, quantify, reorder or learn for THIS JD). No generic advice.
       - skills_to_build: only skills the JD asks for that are genuinely absent from the master profile.
       - interview_points: likely questions for this JD, each tied to one of Syed's real projects or internships.
    6. Keep each project title exactly as it appears in the master profile so links can be matched to it. Do not output any URLs.
    """

    user_prompt = f"""
    MASTER PROFILE JSON:
    {json.dumps(profile_data)}

    USER EXTRA NOTES / CUSTOM OVERRIDES:
    {req.extra_notes}

    TARGET JOB DESCRIPTION (JD):
    {req.jd_text}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=TailoredResumeResponse,
                temperature=0.25,
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

    try:
        tailored = json.loads(response.text)
    except (TypeError, json.JSONDecodeError):
        raise HTTPException(status_code=502, detail="Gemini returned an invalid response. Try again.")

    links_found = _attach_links(tailored, profile_data)

    personal_info = profile_data["personal_info"].copy()
    personal_info["linkedin"] = req.linkedin_url
    personal_info["location"] = req.location

    return {
        "personal_info": personal_info,
        "education": profile_data["education"],
        "achievements": profile_data["achievements"],
        "tailored": tailored,
        "links_found": links_found,
    }


# ---------- job link reader ----------
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _html_to_text(markup: str) -> str:
    parser = _TextExtractor()
    parser.feed(markup)
    parser.close()
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in "".join(parser.parts).split("\n")]
    out, blank = [], False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _jobposting_from_jsonld(page: str) -> str:
    """Many job pages embed the full posting as structured data. Prefer it when present."""
    pattern = r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
    for m in re.finditer(pattern, page, re.S | re.I):
        try:
            data = json.loads(m.group(1).strip())
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        flat = []
        for it in items:
            if isinstance(it, dict) and "@graph" in it:
                flat.extend(it["@graph"])
            else:
                flat.append(it)
        for it in flat:
            if isinstance(it, dict) and "JobPosting" in str(it.get("@type", "")):
                desc = it.get("description", "") or ""
                if "&lt;" in desc:
                    desc = html_lib.unescape(desc)
                desc = _html_to_text(desc)
                org = it.get("hiringOrganization")
                company = org.get("name", "") if isinstance(org, dict) else ""
                if desc:
                    return f"{it.get('title', '')}\n{company}\n\n{desc}".strip()
    return ""


def _assert_public_host(host: str):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not find that website. Check the link.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise HTTPException(status_code=400, detail="That link points to a private address.")


@app.post("/api/fetch-jd")
def fetch_jd(req: FetchJDRequest):
    url = req.url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Paste a full link starting with https://")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    current = url
    try:
        with httpx.Client(timeout=15, follow_redirects=False, headers=headers) as client:
            for _ in range(4):
                p = urlparse(current)
                if p.scheme not in ("http", "https") or not p.hostname:
                    raise HTTPException(status_code=400, detail="That link redirects somewhere unsupported.")
                _assert_public_host(p.hostname)
                resp = client.get(current)
                if resp.status_code in (301, 302, 303, 307, 308) and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                break
            else:
                raise HTTPException(status_code=400, detail="That link redirects too many times.")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Could not open that link: {e}")

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=422,
            detail="This site blocks automatic reading or needs a login. Open the job, copy the description and paste it into the box.",
        )

    page = resp.text[:2_000_000]
    text = _jobposting_from_jsonld(page) or _html_to_text(page)
    if len(text) < 200:
        raise HTTPException(
            status_code=422,
            detail="Could not find a job description on that page (it may load with JavaScript or need a login). Paste it into the box instead.",
        )

    text = text[:12000]
    emails = []
    for e in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        if not re.search(r"no-?reply|donotreply", e, re.I) and e.lower() not in emails:
            emails.append(e.lower())

    return {"text": text, "emails": emails[:3], "final_url": current}


# ---------- email ----------
def _close(server):
    try:
        server.quit()
    except Exception:
        pass


def _smtp_open():
    """Connect and log in. Tries the configured port first, then the other common one."""
    ports = [SMTP_PORT] + [p for p in (465, 587) if p != SMTP_PORT]
    last_error = None
    for port in ports:
        try:
            if port == 465:
                server = smtplib.SMTP_SSL(SMTP_HOST, port, timeout=20)
            else:
                server = smtplib.SMTP(SMTP_HOST, port, timeout=20)
                server.ehlo()
                server.starttls()
                server.ehlo()
            try:
                server.login(SMTP_USER, SMTP_PASSWORD)
            except Exception:
                _close(server)
                raise
            return server
        except smtplib.SMTPAuthenticationError:
            raise  # wrong credentials: another port will not help
        except (OSError, smtplib.SMTPException) as e:
            last_error = e
    raise ConnectionError(str(last_error) if last_error else "unknown network error")


@app.get("/api/email-check")
def email_check():
    if not (SMTP_USER and SMTP_PASSWORD):
        return {"ok": False, "message": "SMTP_USER or SMTP_APP_PASSWORD is missing in api/.env. Add both and restart the server."}
    try:
        server = _smtp_open()
        _close(server)
        return {"ok": True, "message": f"Connected and logged in as {SMTP_USER}. Sending should work."}
    except smtplib.SMTPAuthenticationError:
        return {"ok": False, "message": AUTH_HELP}
    except OSError as e:
        return {"ok": False, "message": f"{NET_HELP} (Details: {e})"}


@app.post("/api/send-email")
def send_email(req: SendEmailRequest):
    if not (SMTP_USER and SMTP_PASSWORD):
        raise HTTPException(
            status_code=503,
            detail="Email sending is not set up. Add SMTP_USER and SMTP_APP_PASSWORD to api/.env, or use Open in Gmail.",
        )

    to_email = req.to_email.strip()
    if not EMAIL_RE.match(to_email):
        raise HTTPException(status_code=400, detail="Recruiter email address is not valid.")

    subject = " ".join(req.subject.split())  # collapses newlines, blocks header injection

    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to_email
    msg["Reply-To"] = SMTP_USER
    msg["Subject"] = subject
    msg.set_content(req.body)

    if req.attachment_b64:
        try:
            raw = base64.b64decode(req.attachment_b64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="Attachment is not valid base64.")
        if len(raw) > MAX_ATTACHMENT_BYTES:
            raise HTTPException(status_code=413, detail="Attachment is larger than 5 MB.")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", req.attachment_name)[:80] or "Resume.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        msg.add_attachment(raw, maintype="application", subtype="pdf", filename=name)

    try:
        server = _smtp_open()
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=502, detail=AUTH_HELP)
    except OSError as e:
        raise HTTPException(status_code=502, detail=f"{NET_HELP} (Details: {e})")

    try:
        server.send_message(msg)
    except smtplib.SMTPRecipientsRefused:
        raise HTTPException(status_code=400, detail="The mail server refused the recruiter's address. Check it for typos.")
    except smtplib.SMTPException as e:
        raise HTTPException(status_code=502, detail=f"The mail server rejected the message: {e}")
    finally:
        _close(server)

    return {"status": "sent", "to": to_email}