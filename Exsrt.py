import streamlit as st
import google.generativeai as genai
from groq import Groq
import re
import time
import zipfile
import io

st.set_page_config(page_title="Gemini SRT Translator", page_icon="🎬", layout="centered")
st.title("🎬 Gemini SRT Subtitle Translator")
st.write("အင်္ဂလိပ် SRT ဖိုင်ကို ထည့်သွင်းပြီး AI သုံး၍ မြန်မာလို ဆီလျော်စွာ ပြန်ဆိုပါ")

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

PROVIDERS = {
    "Gemini (Google)": "gemini",
    "Groq (Llama)": "groq",
}
provider_label = st.radio(
    "🔑 API Provider ရွေးချယ်ပါ (Gemini limit ပြည့်ပါက Groq ကို ပြောင်းသုံးနိုင်သည်)",
    list(PROVIDERS.keys()),
    horizontal=True,
    key="api_provider",
)
provider = PROVIDERS[provider_label]

CHUNK_SIZE = 50


def normalize_newlines(content):
    return content.replace('\r\n', '\n').replace('\r', '\n')


def parse_srt(content):
    content = normalize_newlines(content)
    blocks = re.split(r'\n{2,}', content.strip())
    return [b.strip() for b in blocks if b.strip()]


def clean_ai_output(text):
    # Remove markdown code fences the AI sometimes adds
    text = re.sub(r'^```[^\n]*\n', '', text.strip())
    text = re.sub(r'\n```$', '', text.strip())
    return text.strip()


def build_translate_prompt(blocks):
    chunk_text = "\n\n".join(blocks)
    return (
        "You are a professional movie subtitle translator. "
        "Translate the following English SRT subtitle blocks into Burmese (Myanmar Language) naturally and contextually. "
        "CRITICAL: Keep all subtitle numbers and timestamps (e.g., 00:02:50,904 --> 00:02:52,929) exactly as they are. "
        "Only translate the actual spoken text lines into Burmese. "
        "Return ONLY the translated SRT blocks in the same format, nothing else.\n\n"
        f"{chunk_text}"
    )


def translate_chunk_gemini(model, blocks):
    prompt = build_translate_prompt(blocks)
    response = model.generate_content(prompt)
    return clean_ai_output(response.text)


def translate_chunk_groq(client, blocks):
    prompt = build_translate_prompt(blocks)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return clean_ai_output(response.choices[0].message.content)


TIMESTAMP_LINE = re.compile(
    r'(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})'
)


def normalize_timestamp(ts):
    ts = ts.replace('.', ',')
    parts = re.match(r'(\d{1,2}):(\d{2}):(\d{2}),(\d{1,3})', ts)
    if not parts:
        return ts
    h, m, s, ms = parts.groups()
    ms = ms.ljust(3, '0')[:3]
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms}"


def ts_to_ms(ts):
    parts = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', ts)
    if not parts:
        return 0
    h, m, s, ms = parts.groups()
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)


def ms_to_ts(total_ms):
    total_ms = max(0, total_ms)
    h = total_ms // 3600000
    total_ms %= 3600000
    m = total_ms // 60000
    total_ms %= 60000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


MAX_SUBTITLE_DURATION_MS = 10_000


def fix_srt_timestamps(content):
    content = normalize_newlines(content)
    blocks = parse_srt(content)
    fixed_blocks = []
    issues = []
    prev_end_ms = 0

    for idx, block in enumerate(blocks):
        lines = block.splitlines()
        fixed_lines = []

        for line in lines:
            match = TIMESTAMP_LINE.match(line.strip())
            if match:
                original = line.strip()
                start = normalize_timestamp(match.group(1))
                end = normalize_timestamp(match.group(2))
                reasons = []

                if '.' in match.group(1) or '.' in match.group(2):
                    reasons.append("dot→comma")
                if match.group(1).startswith('0:') or match.group(2).startswith('0:'):
                    reasons.append("leading zero")
                ms1 = re.split(r'[,.]', match.group(1))[-1]
                ms2 = re.split(r'[,.]', match.group(2))[-1]
                if len(ms1) < 3 or len(ms2) < 3:
                    reasons.append("short ms")

                start_ms = ts_to_ms(start)
                end_ms = ts_to_ms(end)

                if start_ms < prev_end_ms:
                    fixed_start_ms = prev_end_ms + 100
                    start = ms_to_ts(fixed_start_ms)
                    start_ms = fixed_start_ms
                    reasons.append("out-of-order start")

                duration_ms = end_ms - start_ms
                if duration_ms > MAX_SUBTITLE_DURATION_MS:
                    fixed_end_ms = start_ms + 5000  # cap at 5 seconds
                    end = ms_to_ts(fixed_end_ms)
                    end_ms = fixed_end_ms
                    reasons.append(f"too long ({duration_ms//1000}s→5s)")

                fixed_line = f"{start} --> {end}"
                if fixed_line != original or reasons:
                    issues.append({
                        "block": idx + 1,
                        "before": original,
                        "after": fixed_line,
                        "reason": ", ".join(reasons) if reasons else "format",
                    })

                prev_end_ms = end_ms
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        renumbered = str(idx + 1)
        if fixed_lines and fixed_lines[0].strip().isdigit():
            if fixed_lines[0].strip() != renumbered:
                issues.append({
                    "block": idx + 1,
                    "before": f"Block number: {fixed_lines[0].strip()}",
                    "after": f"Block number: {renumbered}",
                    "reason": "wrong number",
                })
            fixed_lines[0] = renumbered
        else:
            fixed_lines.insert(0, renumbered)

        fixed_blocks.append("\n".join(fixed_lines))

    return "\n\n".join(fixed_blocks), issues


def show_ts_issues(issues):
    st.warning(f"Timestamp ပြဿနာ {len(issues)} ခု တွေ့ရှိပြီး အလိုအလျောက် ပြုပြင်ပြီးစီးသည်:")
    col_block, col_before, col_after, col_reason = st.columns([1, 3, 3, 2])
    col_block.markdown("**Block**")
    col_before.markdown("**မူရင်း (Before)**")
    col_after.markdown("**ပြင်ဆင်ပြီး (After)**")
    col_reason.markdown("**အကြောင်းရင်း**")
    st.markdown("---")
    for item in issues:
        c1, c2, c3, c4 = st.columns([1, 3, 3, 2])
        c1.markdown(f"`{item['block']}`")
        c2.markdown(f"~~`{item['before']}`~~")
        c3.markdown(f"`{item['after']}`")
        c4.markdown(f"_{item['reason']}_")


# --- Tabs ---
tab1, tab2 = st.tabs(["🌐 Translate", "🔧 Fix Timestamps"])

# ── Tab 1: Translate ──────────────────────────────────────────────
with tab1:
    uploaded_files = st.file_uploader(
        "English SRT ဖိုင်များကို ရွေးချယ်ပါ (တစ်ကြိမ်တည်း အများအပြား ရွေးနိုင်သည်)",
        type=["srt"],
        accept_multiple_files=True,
        key="translate_upload"
    )

    if uploaded_files:
        st.info(f"ဖိုင် {len(uploaded_files)} ဖိုင် ရွေးချယ်ထားသည်။")

        if st.button("Translate & Export MM SRT", type="primary"):
            if provider == "gemini":
                if not GEMINI_API_KEY:
                    st.error("GEMINI_API_KEY ကို secrets.toml တွင် သတ်မှတ်ထားခြင်း မရှိပါ။")
                    st.stop()
                model = genai.GenerativeModel("gemini-2.5-flash")
            else:
                if not GROQ_API_KEY:
                    st.error("GROQ_API_KEY ကို secrets.toml တွင် သတ်မှတ်ထားခြင်း မရှိပါ။")
                    st.stop()
                groq_client = Groq(api_key=GROQ_API_KEY)
            st.session_state["translate_results"] = []

            for file_idx, uploaded_file in enumerate(uploaded_files):
                st.markdown(f"**[{file_idx+1}/{len(uploaded_files)}] ဘာသာပြန်နေသည်: `{uploaded_file.name}`**")
                srt_content = uploaded_file.read().decode("utf-8")
                blocks = parse_srt(srt_content)
                total_blocks = len(blocks)
                chunks = [blocks[i:i + CHUNK_SIZE] for i in range(0, total_blocks, CHUNK_SIZE)]
                progress_bar = st.progress(0)
                status_text = st.empty()
                translated_blocks = []
                try:
                    for i, chunk in enumerate(chunks):
                        status_text.text(f"({min((i+1)*CHUNK_SIZE, total_blocks)}/{total_blocks} subtitles)")
                        if provider == "gemini":
                            result = translate_chunk_gemini(model, chunk)
                        else:
                            result = translate_chunk_groq(groq_client, chunk)
                        translated_blocks.append(result)
                        progress_bar.progress((i + 1) / len(chunks))
                        time.sleep(0.3)
                    translated_srt = "\n\n".join(translated_blocks)
                    status_text.text("Timestamp စစ်ဆေးနေသည်...")
                    translated_srt, ts_issues = fix_srt_timestamps(translated_srt)
                    status_text.text("✅ ပြီးစီးသည်!")
                    progress_bar.progress(1.0)
                    st.session_state["translate_results"].append({
                        "name": uploaded_file.name,
                        "srt": translated_srt.encode("utf-8"),
                        "ts_issues": ts_issues,
                        "error": None,
                    })
                except Exception as e:
                    status_text.text("❌ အမှားဖြစ်သည်။")
                    st.session_state["translate_results"].append({
                        "name": uploaded_file.name, "srt": None, "ts_issues": [], "error": str(e)
                    })

        if st.session_state.get("translate_results"):
            results = st.session_state["translate_results"]
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    if r["srt"]:
                        zf.writestr(r["name"], r["srt"])
            zip_buffer.seek(0)

            st.divider()
            st.download_button(
                label=f"📦 Download All ({len([r for r in results if r['srt']])} files) — ZIP",
                data=zip_buffer,
                file_name="translated_subtitles.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_translate_all",
            )
            st.caption("ZIP ကို ဖြည်၍ မူရင်းဖိုင်တွဲတွင် ထည့်သောအခါ ဖိုင်များ အစားထိုးမည်ဖြစ်သည်။")
            for file_idx, r in enumerate(results):
                st.divider()
                st.markdown(f"### [{file_idx+1}] `{r['name']}`")
                if r["error"]:
                    st.error(f"Error: {r['error']}")
                else:
                    if r["ts_issues"]:
                        show_ts_issues(r["ts_issues"])
                    else:
                        st.success("Timestamp အားလုံး မှန်ကန်သည်။")
                    st.download_button(
                        label=f"📥 Download: {r['name']}",
                        data=r["srt"],
                        file_name=r["name"],
                        mime="text/plain",
                        key=f"dl_translate_{file_idx}",
                    )

# ── Tab 2: Fix Timestamps ─────────────────────────────────────────
with tab2:
    st.subheader("🔧 Fix Strange Timestamps")
    st.write("Export ပြုလုပ်ပြီးသော SRT ဖိုင်၏ မှားယွင်းနေသော timestamp များကို အလိုအလျောက် ပြုပြင်ပေးသည်။")

    fix_files = st.file_uploader(
        "SRT ဖိုင်များကို ရွေးချယ်ပါ (တစ်ကြိမ်တည်း အများအပြား ရွေးနိုင်သည်)",
        type=["srt"],
        accept_multiple_files=True,
        key="fix_upload"
    )

    if fix_files:
        st.info(f"ဖိုင် {len(fix_files)} ဖိုင် ရွေးချယ်ထားသည်။")

        if st.button("Fix Timestamps", type="primary"):
            st.session_state["fix_results"] = []
            for fix_file in fix_files:
                raw = fix_file.read().decode("utf-8")
                fixed_srt, issues = fix_srt_timestamps(raw)
                st.session_state["fix_results"].append({
                    "name": fix_file.name,
                    "srt": fixed_srt.encode("utf-8"),
                    "issues": issues,
                })

        if st.session_state.get("fix_results"):
            results = st.session_state["fix_results"]
            total_issues = sum(len(r["issues"]) for r in results)

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    zf.writestr(r["name"], r["srt"])
            zip_buffer.seek(0)

            st.divider()
            all_label = f"📦 Download All ({len(results)} files) — ZIP"
            if total_issues:
                all_label += f"  •  {total_issues} fixes applied"
            st.download_button(
                label=all_label,
                data=zip_buffer,
                file_name="fixed_subtitles.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_fix_all",
            )
            st.caption("ZIP ကို ဖြည်၍ မူရင်းဖိုင်တွဲတွင် ထည့်သောအခါ ဖိုင်များ အစားထိုးမည်ဖြစ်သည်။")

            for file_idx, r in enumerate(results):
                st.divider()
                st.markdown(f"### [{file_idx+1}] `{r['name']}`")
                if r["issues"]:
                    show_ts_issues(r["issues"])
                else:
                    st.success("Timestamp အားလုံး မှန်ကန်နေသည်။ ပြုပြင်ရန် မလိုအပ်ပါ။")
                st.download_button(
                    label=f"📥 Download: {r['name']}",
                    data=r["srt"],
                    file_name=r["name"],
                    mime="text/plain",
                    key=f"dl_fix_{file_idx}",
                )
