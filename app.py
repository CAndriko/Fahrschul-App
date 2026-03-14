import streamlit as st
import json
import os
from datetime import datetime
from openai import OpenAI

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & DATA STORAGE
# -----------------------------------------------------------------------------
DATA_FILE = "student_data.json"

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {"students": {}}

def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def format_phone_number(phone: str) -> str:
    cleaned = ''.join([c for c in phone if c.isdigit()])
    if not cleaned:
        return ""
    if cleaned.startswith("00"):
        cleaned = cleaned[2:]
    elif cleaned.startswith("0"):
        cleaned = "49" + cleaned[1:]
    return cleaned

# NEU: Funktion, die das Archiv in einen sauberen Text für den Download verwandelt
def generate_export_text(student_name: str, logs: list) -> str:
    text = f"FAHRSCHUL-AKTE: {student_name}\n"
    text += "="*50 + "\n\n"
    for log in logs:
        text += f"Datum: {log['date']}\n"
        text += "-"*50 + "\n"
        text += f"WhatsApp-Nachricht:\n{log['whatsapp_msg']}\n\n"
        text += "Ampel-Logbuch:\n"
        for item in log['logbook']:
            text += f"{item.get('status', '')} {item.get('category', '')}: {item.get('note', '')}\n"
        text += "\n" + "="*50 + "\n\n"
    return text

# -----------------------------------------------------------------------------
# 2. AI LOGIC (Whisper + GPT-4o-mini)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_driving_lesson(audio_bytes: bytes, api_key: str, student_name: str) -> dict:
    try:
        client = OpenAI(api_key=api_key)
        
        temp_file = "temp_recording.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
            
        with open(temp_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                prompt="Fahrstunde, Fahrschule, Fahrlehrer, einparken, ausparken, Vorfahrt, Schulterblick, abwürgen, kuppeln, schalten, Spiegel, Blinker, rechts vor links, überholen, bremsen, Fußgänger."
            )
            
        spoken_text = transcript.text
        
        prompt = f"""
        Du bist ein professioneller Assistent für einen Fahrlehrer.
        Analysiere das folgende diktierte Protokoll einer Fahrstunde für den Schüler '{student_name}'.
        Ignoriere irrelevantes Gerede.
        
        Erstelle ein JSON-Objekt mit exakt diesen zwei Schlüsseln:
        1. "whatsapp_msg": Eine freundliche WhatsApp-Nachricht an den Schüler als kurze Zusammenfassung.
        2. "logbook": Eine Liste von Objekten für das interne Ampel-System. Jedes Objekt hat:
           - "status": Ein Emoji ("🟢", "🟡", "🔴")
           - "category": Kurze Kategorie (z.B. "Einparken", "Vorfahrt", "Schalten")
           - "note": Kurze, sachliche Notiz für den Fahrlehrer.
           
        Protokoll: {spoken_text}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Assistent, der ausschließlich gültiges JSON ausgibt."},
                {"role": "user", "content": prompt}
            ]
        )
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        return {
            "whatsapp_msg": f"Fehler bei der KI-Analyse: {str(e)}",
            "logbook": [{"status": "🔴", "category": "Systemfehler", "note": "Bitte API-Key prüfen oder Internetverbindung checken."}]
        }

# -----------------------------------------------------------------------------
# 3. USER INTERFACE
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Logbuch Michael", page_icon="🚘", layout="centered")
    
    if "db" not in st.session_state:
        st.session_state.db = load_data()
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None

    with st.sidebar:
        st.title("🚘 Drive & Ride")
        st.subheader("Logbuch Michael")
        st.markdown("---")
        
        st.subheader("👥 Schüler")
        
        with st.expander("➕ Neuen Schüler anlegen"):
            new_name = st.text_input("Name des Schülers")
            new_phone = st.text_input("Handynummer (Ziel der Nachricht)")
            
            if st.button("Speichern", use_container_width=True):
                if new_name and new_name not in st.session_state.db["students"]:
                    formatted_phone = format_phone_number(new_phone)
                    st.session_state.db["students"][new_name] = {"phone": formatted_phone, "logs": []}
                    save_data(st.session_state.db)
                    st.success(f"{new_name} angelegt!")
                    st.rerun()
                elif new_name:
                    st.warning("Schüler existiert bereits.")

        student_list = list(st.session_state.db["students"].keys())
        selected_student = None
        
        if student_list:
            selected_student = st.selectbox("📂 Aktiver Schüler", student_list)
            
            if st.session_state.delete_confirm != selected_student:
                if st.button("🗑️ Schüler löschen", use_container_width=True):
                    st.session_state.delete_confirm = selected_student
                    st.rerun()
                    
            if st.session_state.delete_confirm == selected_student:
                st.error(f"'{selected_student}' inklusive Archiv endgültig löschen?")
                col1, col2 = st.columns(2)
                if col1.button("Ja, weg damit", type="primary", use_container_width=True):
                    del st.session_state.db["students"][selected_student]
                    save_data(st.session_state.db)
                    st.session_state.delete_confirm = None
                    st.rerun()
                if col2.button("Abbrechen", use_container_width=True):
                    st.session_state.delete_confirm = None
                    st.rerun()
        else:
            st.info("Bitte lege zuerst einen Schüler an.")

        st.markdown("---")
        
        with st.expander("⚙️ Einstellungen"):
            api_key = st.text_input("🔑 OpenAI API Key", type="password", value=st.session_state.get("api_key", ""), help="Dein sk-proj... Key")
            st.session_state.api_key = api_key

    st.title("🎙️ Fahrstunde dokumentieren")
    
    if not selected_student:
        st.warning("👈 Wähle links einen Schüler aus oder lege einen neuen an, um zu starten.")
        return

    st.markdown(f"**Aktueller Schüler:** {selected_student}")
    st.markdown("---")
    
    tab_record, tab_archive = st.tabs(["🎙️ Neue Aufnahme", "🗂️ Schüler-Akte (Archiv)"])

    with tab_record:
        audio_value = st.audio_input("Sprich hier dein Protokoll ein")

        if audio_value:
            if not st.session_state.api_key:
                st.error("Bitte trage links unten in den ⚙️ Einstellungen deinen API Key ein.")
            else:
                with st.spinner("KI analysiert die Fahrt..."):
                    results = analyze_driving_lesson(audio_value.getvalue(), st.session_state.api_key, selected_student)
                    
                    st.markdown("### 📱 WhatsApp Vorschau")
                    st.info(results["whatsapp_msg"])
                    
                    phone = st.session_state.db["students"][selected_student].get("phone", "")
                    if phone:
                        encoded_msg = results.get("whatsapp_msg", "").replace(" ", "%20").replace("\n", "%0A")
                        whatsapp_url = f"https://wa.me/{phone}?text={encoded_msg}"
                        st.link_button("Nachricht in WhatsApp öffnen & senden", whatsapp_url, type="primary")
                    
                    st.markdown("---")
                    
                    st.markdown("### 🚦 Internes Ampel-Logbuch")
                    for item in results.get("logbook", []):
                        st.markdown(f"**{item.get('status', '⚪')} {item.get('category', 'Notiz')}** \n*{item.get('note', '')}*")
                        
                    if st.button("💾 Logbuch final in die Akte speichern", type="primary"):
                        new_log = {
                            "date": datetime.now().strftime("%d.%m.%Y um %H:%M Uhr"),
                            "whatsapp_msg": results.get("whatsapp_msg", ""),
                            "logbook": results.get("logbook", [])
                        }
                        st.session_state.db["students"][selected_student]["logs"].insert(0, new_log)
                        save_data(st.session_state.db)
                        
                        st.success("Erfolgreich im Archiv gespeichert! Schau in den Tab 'Schüler-Akte'.")

    with tab_archive:
        logs = st.session_state.db["students"][selected_student].get("logs", [])
        
        if not logs:
            st.info(f"Das Archiv von {selected_student} ist noch leer.")
        else:
            # NEU: Der Download-Button für das Archiv
            export_data = generate_export_text(selected_student, logs)
            st.download_button(
                label=f"📄 Komplette Akte von {selected_student} herunterladen",
                data=export_data,
                file_name=f"Fahrschul_Akte_{selected_student.replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.markdown("---")
            
            for i, log in enumerate(logs):
                with st.expander(f"🗓️ Fahrstunde vom {log['date']}"):
                    st.markdown("**WhatsApp-Nachricht:**")
                    st.write(log["whatsapp_msg"])
                    st.markdown("**Ampel-Logbuch:**")
                    for item in log["logbook"]:
                        st.markdown(f"**{item.get('status', '⚪')} {item.get('category', 'Notiz')}**: {item.get('note', '')}")

if __name__ == "__main__":
    main()