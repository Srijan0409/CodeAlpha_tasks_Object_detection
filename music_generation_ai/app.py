import os
import sys
import time
import pandas as pd
import streamlit as st
import subprocess

# Set page configuration with standard options
st.set_page_config(
    page_title="Cosmo Music AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SECTION 1 — COLOUR PALETTE & VISUAL IDENTITY / SECTION 4 — ANIMATION & VISUAL POLISH
# Deep space navy background, supernova accents, custom CSS slider/pill overrides, star particle animation
st.markdown("""
<style>
/* Global App Background & Base Styles */
.stApp { background: #000014 !important; color: #F5F0E8; }
[data-testid="stSidebar"] { background: #0e0b1e !important; border-right: 1px solid rgba(255,107,53,0.12); }
h1,h2,h3 { color: #F5F0E8 !important; }

/* Sliders */
.stSlider > div > div > div { background: linear-gradient(90deg,#FF6B35,#C850C0) !important; }
.stSlider [data-testid="stThumbValue"] { color: #FF8C5A !important; }

/* Primary CTA Button Overrides */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg,#FF6B35,#C850C0,#FF6B35) !important;
  background-size: 200% !important;
  border: none !important; color: white !important; font-weight: 700 !important;
  border-radius: 14px !important; padding: 14px 28px !important; font-size: 15px !important;
  letter-spacing: 0.04em !important; transition: all 0.3s !important;
  animation: shimmer 3s linear infinite, morphBtn 4s ease infinite !important;
}
div[data-testid="stButton"] > button:hover { transform: scale(1.03) !important; filter: brightness(1.1) !important; }

/* Selectbox and Radio */
.stSelectbox > div, .stRadio > div { color: #8B7FA8 !important; }

/* Metric Cards */
.stMetric { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; padding: 16px !important; transition: all 0.3s !important; }
.stMetric:hover { border-color: rgba(255,107,53,0.35) !important; transform: translateY(-3px) !important; }
.stMetric label { color: #5c5070 !important; font-size: 10px !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
.stMetric [data-testid="stMetricValue"] { color: #FF8C5A !important; font-size: 22px !important; font-weight: 700 !important; }

/* Download Buttons */
[data-testid="stDownloadButton"] > button {
  background: rgba(255,107,53,0.12) !important; border: 1px solid rgba(255,107,53,0.3) !important;
  color: #FF8C5A !important; border-radius: 10px !important; font-weight: 600 !important;
  transition: all 0.25s !important;
}
[data-testid="stDownloadButton"] > button:hover { transform: translateY(-2px) !important; filter: brightness(1.2) !important; }

/* Existing Animations & Utilities */
@keyframes shimmer {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
@keyframes morphBtn {
  0%, 100% { border-radius: 14px; }
  50% { border-radius: 18px; }
}
@keyframes pulseWave1 { 0%, 100% { height: 15%; } 50% { height: 85%; } }
@keyframes pulseWave2 { 0%, 100% { height: 35%; } 50% { height: 100%; } }
@keyframes pulseWave3 { 0%, 100% { height: 10%; } 50% { height: 65%; } }
@keyframes pulseWave4 { 0%, 100% { height: 55%; } 50% { height: 95%; } }
@keyframes pulseWave5 { 0%, 100% { height: 25%; } 50% { height: 75%; } }
@keyframes twinkle {
    0% { opacity: 0.15; }
    50% { opacity: 0.75; }
    100% { opacity: 0.15; }
}

/* Custom Card styling classes (retained for custom containers) */
.cosmic-card {
    background-color: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.tip-box {
    background-color: #0e0b1e;
    border: 1px solid #9B59B6;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    margin-bottom: 15px;
}
.title-accent {
    background: linear-gradient(90deg, #FF6B35, #FF9A3C, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    line-height: 1.2;
}
.subtitle-accent {
    background: linear-gradient(90deg, #C850C0, #9B59B6, #6C63FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem;
    line-height: 1.2;
}
.badge-pill {
    background-color: rgba(255,255,255,0.03);
    color: #4DD9AC;
    padding: 4px 14px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 12px;
    border: 1px solid #4DD9AC;
}
.star-particles {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    z-index: 0;
    pointer-events: none;
    background-image: 
        radial-gradient(2px 2px at 25px 35px, #F5F0E8, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 50px 80px, #8B7FA8, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 80px 170px, #F5F0E8, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 120px 45px, #F5F0E8, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 160px 95px, #8B7FA8, rgba(0,0,0,0)),
        radial-gradient(2px 2px at 190px 140px, #F5F0E8, rgba(0,0,0,0));
    background-repeat: repeat;
    background-size: 250px 250px;
    animation: twinkle 6s infinite;
    opacity: 0.35;
}
.secondary-text {
    color: #8B7FA8 !important;
    font-size: 0.95rem;
}
.hint-text {
    color: #4a3d6b !important;
    font-size: 0.8rem;
}
</style>
<!-- Star Particles Anchor -->
<div class="star-particles"></div>
""", unsafe_allow_html=True)

# Initialize global Session State tracking variables safely
if "total_notes_generated" not in st.session_state:
    st.session_state.total_notes_generated = 0
if "last_track_duration" not in st.session_state:
    st.session_state.last_track_duration = "N/A"
if "generated_tracks_count" not in st.session_state:
    st.session_state.generated_tracks_count = 0
if "has_generated" not in st.session_state:
    st.session_state.has_generated = False
if "first_load" not in st.session_state:
    st.session_state.first_load = True

# Resolve Project workspace metrics
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "music_model.h5")
model_exists = os.path.exists(model_path)

# Scan target repository folders for metadata mapping
midi_dir = os.path.join(base_dir, "midi_data")
midi_files_count = 0
if os.path.exists(midi_dir):
    try:
        midi_files_count = len([f for f in os.listdir(midi_dir) if f.endswith(".mid")])
    except Exception:
        midi_files_count = 0

loss_csv = os.path.join(base_dir, "loss_history.csv")
epochs_count = 0
if os.path.exists(loss_csv):
    try:
        df_loss = pd.read_csv(loss_csv)
        epochs_count = len(df_loss)
    except Exception:
        epochs_count = 100 if model_exists else 0
else:
    epochs_count = 100 if model_exists else 0

# SECTION 2 — PAGE LAYOUT & STRUCTURE: LEFT SIDEBAR
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 25px;">
    <span style="font-size: 1.8rem; color: #FF6B35;">✦</span>
    <span style="font-size: 1.4rem; font-weight: 800; background: linear-gradient(90deg, #FF6B35, #FF9A3C); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Cosmo Music AI</span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown('<p style="font-size: 0.75rem; color: #4a3d6b; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 5px;">NAVIGATE</p>', unsafe_allow_html=True)

page_selection = st.sidebar.radio(
    label="Navigation Selection",
    options=["✦ Generate", "✦ Training Monitor", "✦ My Tracks"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr style='border-color: #1e1535; margin: 15px 0;'>", unsafe_allow_html=True)

# Model Status Info Box
status_dot = "🟢" if model_exists else "🔴"
status_text = "Model ready" if model_exists else "Model not trained yet"
status_color = "#4DD9AC" if model_exists else "#FF6B35"

st.sidebar.markdown(f"""
<div class="cosmic-card" style="padding: 16px; margin-bottom: 0;">
    <p style="font-size: 0.85rem; color: #8B7FA8; margin-bottom: 10px; font-weight: 700; letter-spacing: 0.5px;">MODEL STATUS</p>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
        <span style="font-size: 0.9rem;">{status_dot}</span>
        <span style="color: {status_color}; font-weight: 700; font-size: 0.95rem;">{status_text}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #8B7FA8; margin-bottom: 6px;">
        <span>Trained Epochs:</span>
        <span style="color: #F5F0E8; font-weight: 600;">{epochs_count}</span>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #8B7FA8;">
        <span>Source MIDI Files:</span>
        <span style="color: #F5F0E8; font-weight: 600;">{midi_files_count}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# MAIN CONTENT AREA INTEGRATION
if page_selection == "✦ Generate":
    # SECTION 3 — BEGINNER-FRIENDLINESS RULES: Welcome banner on initial session access
    if st.session_state.first_load:
        st.markdown("""
        <div style="background-color: #110e1f; border: 1px solid #1e1535; border-left: 4px solid #C850C0; padding: 14px 20px; border-radius: 8px; margin-bottom: 25px;">
            <p style="margin: 0; color: #F5F0E8; font-weight: 500; font-size: 0.95rem;">
                ✦ <b>Welcome to Cosmo Music AI!</b> Start by clicking Generate to create your first track. No experience needed.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Row 1: Hero section
    st.markdown('<div class="badge-pill">✦ AI Music Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="title-accent">Create music from stardust</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-accent">& imagination</div>', unsafe_allow_html=True)
    st.markdown('<p class="secondary-text" style="font-size: 1.15rem; margin-top: 12px; margin-bottom: 35px;">No music knowledge needed. Pick a style, set the mood, and let the AI compose for you.</p>', unsafe_allow_html=True)
    
    # Pre-resolve Creativity tier string mapping for accurate inline stat presentation
    curr_creat = st.session_state.get("creativity_slider", 1.0)
    if curr_creat < 0.7:
        creat_level = "Low"
    elif curr_creat <= 1.2:
        creat_level = "Medium"
    elif curr_creat <= 1.7:
        creat_level = "High"
    else:
        creat_level = "Cosmic"
        
    # Row 2: 3 stat cards side by side
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Notes Generated", st.session_state.total_notes_generated)
    with col2:
        st.metric("Current Creativity Level", creat_level)
    with col3:
        st.metric("Estimated Last Track Duration", st.session_state.last_track_duration)
        
    # Row 3: Compose Settings card
    st.markdown("### ✦ Compose Settings")
    
    with st.container():
        # Creativity Slider
        creat_val = st.slider("Creativity Mode", 0.3, 2.0, 1.0, 0.1, key="creativity_slider")
        
        # Calculate sub-labels per specification guidelines
        if creat_val < 0.7:
            creat_desc = "Safe & melodic"
            tip_text = "Great for background music and relaxing vibes."
        elif creat_val <= 1.2:
            creat_desc = "Balanced"
            tip_text = "Balanced — sounds musical but with creative surprises."
        elif creat_val <= 1.7:
            creat_desc = "Wild"
            tip_text = "Expect the unexpected. Some notes will surprise you!"
        else:
            creat_desc = "Cosmic"
            tip_text = "Fully experimental. This is alien music territory."
            
        st.markdown(f'<p class="secondary-text" style="margin-top: -12px; margin-bottom: 25px;">Current style profile: <b>{creat_desc}</b>. Adjusts model unpredictability during note sequencing.</p>', unsafe_allow_html=True)
        
        # Length Slider
        length_val = st.slider("Length (notes)", 100, 1000, 500, 50, key="length_slider")
        total_seconds = int(length_val * 0.25)
        mins = total_seconds // 60
        secs = total_seconds % 60
        duration_str = f"~{mins} minutes {secs} seconds" if mins > 0 else f"~{secs} seconds"
        
        st.markdown(f'<p class="secondary-text" style="margin-top: -12px; margin-bottom: 25px;">Calculated Playback Duration: <b>{duration_str}</b>. Set the raw sequence count composing your entire track.</p>', unsafe_allow_html=True)
        
        # Genre selector rendered as pills
        st.markdown('<p style="font-size: 1rem; font-weight: 600; margin-bottom: 6px; color: #F5F0E8;">Musical Genre Setting</p>', unsafe_allow_html=True)
        genre_val = st.radio(
            label="Genre selector",
            options=["Classical", "Jazz", "Ambient", "Blues", "Cinematic"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown('<p class="secondary-text">Infuse specific melodic structures and chord styling rules into the final generated stream.</p>', unsafe_allow_html=True)
        
        # Context-aware beginner tip box
        st.markdown(f"""
        <div class="tip-box">
            <p style="color: #9B59B6; font-size: 0.85rem; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.5px;">✦ BEGINNER TIP</p>
            <p style="color: #F5F0E8; font-size: 0.95rem; margin: 0;">{tip_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Row 4: Generate button logic
    button_tooltip = "Compose your deep learning composition using target specifications." if model_exists else "Train your model first. Go to Training Monitor → Re-train model."
    
    generate_clicked = st.button("✦ Generate my music", disabled=not model_exists, help=button_tooltip, use_container_width=True)
    
    if generate_clicked:
        # Wrap execution safely inside resilient try/except catching handles
        try:
            os.environ["TEMPERATURE"] = str(creat_val)
            os.environ["NOTE_COUNT"] = str(length_val)
            
            with st.spinner("Composing your track from stardust... Please wait while neural weights sequence your audio."):
                script_path = os.path.join(base_dir, "generate.py")
                res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
                
            if res.returncode == 0:
                st.session_state.has_generated = True
                st.session_state.generated_tracks_count += 1
                st.session_state.total_notes_generated += length_val
                st.session_state.last_track_duration = duration_str
                st.session_state.first_load = False
                
                # Rule 5: show confetti balloons
                st.balloons()
                # Rule 6: show non-blocking success toast
                st.toast("✦ Track successfully generated!", icon="✨")
                
                st.success("✦ Success! Your space composition is complete and prepared for download.", icon="⭐")
            else:
                # Rule 1: Guarding raw errors with beginner context messages
                st.error("Something went wrong — make sure your model weights are trained properly or check your target dependencies.")
        except Exception as e:
            st.error("Something went wrong — make sure your model is trained first.")
            
    # Row 5: Output player display component
    if st.session_state.has_generated:
        st.markdown("<hr style='border-color: #1e1535; margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("### ✦ Output Player")
        
        output_mid_path = os.path.join(base_dir, "output.mid")
        output_wav_path = os.path.join(base_dir, "output.wav")
        
        track_filename = f"cosmic_melody_{st.session_state.generated_tracks_count:03d}.mid"
        
        st.markdown(f"""
        <div class="cosmic-card" style="border-color: #C850C0; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #F5F0E8; font-size: 1.1rem;">🎵 {track_filename}</span>
                <span style="background-color: #1e1535; color: #FF9A3C; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: 600;">Synthesized Track</span>
            </div>
            
            <!-- Animated Waveform Visualizer -->
            <div style="display: flex; align-items: flex-end; justify-content: center; gap: 6px; height: 100px; margin: 25px 0; background: #0d0d1a; padding: 15px; border-radius: 8px; border: 1px solid #1e1535;">
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave1 0.9s infinite ease-in-out;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave2 1.2s infinite ease-in-out 0.1s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave3 0.7s infinite ease-in-out 0.2s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave4 1.1s infinite ease-in-out 0.3s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave5 0.8s infinite ease-in-out 0.15s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave1 1.3s infinite ease-in-out 0.05s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave2 0.6s infinite ease-in-out 0.4s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave4 1.4s infinite ease-in-out 0.25s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave3 0.9s infinite ease-in-out 0.1s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave5 1.0s infinite ease-in-out 0.35s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave2 1.1s infinite ease-in-out 0.2s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave1 0.7s infinite ease-in-out 0.05s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave4 1.3s infinite ease-in-out 0.3s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave3 0.8s infinite ease-in-out 0.15s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave5 1.2s infinite ease-in-out 0.4s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave1 0.9s infinite ease-in-out 0.25s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave2 1.0s infinite ease-in-out 0.1s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave4 0.6s infinite ease-in-out 0.35s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave3 1.4s infinite ease-in-out 0.05s;"></div>
                <div style="width: 14px; background: linear-gradient(to top, #FF6B35, #C850C0); border-radius: 3px; animation: pulseWave5 0.8s infinite ease-in-out 0.2s;"></div>
            </div>
            
            <!-- Visual playback sequence mapping -->
            <div style="width: 100%; background-color: #1e1535; height: 6px; border-radius: 3px; margin-bottom: 8px; overflow: hidden;">
                <div style="width: 60%; background: linear-gradient(90deg, #FF6B35, #C850C0); height: 100%; border-radius: 3px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #8B7FA8; margin-bottom: 25px;">
                <span>0:00</span>
                <span>Visual audio tracking active</span>
                <span>{st.session_state.last_track_duration}</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Download handles
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            if os.path.exists(output_mid_path):
                try:
                    with open(output_mid_path, "rb") as f:
                        mid_bytes = f.read()
                    st.download_button(
                        label="📥 Download MIDI Sequence",
                        data=mid_bytes,
                        file_name=track_filename,
                        mime="audio/midi",
                        use_container_width=True
                    )
                    st.markdown('<p class="hint-text" style="text-align: center; margin-top: 6px;">Universal MIDI standard format, directly compatible with any production DAW.</p>', unsafe_allow_html=True)
                except Exception:
                    st.error("Failed to load destination artifact output.mid.")
            else:
                st.warning("Generated MIDI file could not be read.")
                
        with d_col2:
            if os.path.exists(output_wav_path):
                try:
                    with open(output_wav_path, "rb") as f:
                        wav_bytes = f.read()
                    wav_filename = track_filename.replace(".mid", ".wav")
                    st.download_button(
                        label="📥 Download WAV Audio",
                        data=wav_bytes,
                        file_name=wav_filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
                    st.markdown('<p class="hint-text" style="text-align: center; margin-top: 6px;">Fully rendered digital WAV recording ready for immediate playback.</p>', unsafe_allow_html=True)
                except Exception:
                    st.error("Failed to load destination audio output.wav.")
            else:
                st.markdown("""
                <div style="background-color: #1e1535; border: 1px solid #4a3d6b; border-radius: 8px; padding: 10px; text-align: center; height: 42px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #8B7FA8; font-size: 0.85rem;">ℹ️ Install <b>FluidSynth</b> for WAV export</span>
                </div>
                <p class="hint-text" style="text-align: center; margin-top: 6px;">FluidSynth synthesizes pure audio using hardware soundfont matrices.</p>
                """, unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)

# PAGE 2: Training Monitor
elif page_selection == "✦ Training Monitor":
    st.markdown('<div class="title-accent">Training Convergence Monitor</div>', unsafe_allow_html=True)
    st.markdown('<p class="secondary-text" style="font-size: 1.1rem; margin-top: 10px; margin-bottom: 30px;">Evaluate network loss regression streams across continuous execution epochs.</p>', unsafe_allow_html=True)
    
    loss_csv = os.path.join(base_dir, "loss_history.csv")
    
    if os.path.exists(loss_csv):
        try:
            df_loss = pd.read_csv(loss_csv)
            if not df_loss.empty and "Loss" in df_loss.columns:
                best_loss = df_loss["Loss"].min()
                last_loss = df_loss["Loss"].iloc[-1]
                total_epochs = len(df_loss)
                
                t_col1, t_col2, t_col3 = st.columns(3)
                with t_col1:
                    st.metric("Epochs Trained", total_epochs)
                with t_col2:
                    st.metric("Optimal Convergence Loss", f"{best_loss:.4f}")
                with t_col3:
                    st.metric("Final Terminal Loss", f"{last_loss:.4f}")
                    
                st.markdown("### ✦ Optimization Loss Lifecycle")
                st.markdown('<p class="secondary-text" style="margin-top: -10px; margin-bottom: 20px;">A descending trajectory confirms standard pattern learning across preprocessed sequence input streams.</p>', unsafe_allow_html=True)
                
                chart_data = df_loss.set_index("Epoch") if "Epoch" in df_loss.columns else df_loss["Loss"]
                try:
                    st.line_chart(chart_data, color="#FF6B35")
                except Exception:
                    st.line_chart(chart_data)
            else:
                st.warning("Stored loss_history.csv format parsing error.")
        except Exception:
            st.error("Something went wrong — make sure your model is trained first.")
    else:
        st.info("✦ Train your model first using python train.py to compile historical validation curves.")
        
    st.markdown("<hr style='border-color: #1e1535; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### ✦ Deep Architecture Core Refresher")
    st.markdown('<p class="secondary-text" style="margin-bottom: 20px;">Execute a complete neural retraining subroutine. The algorithm extracts target pitchnames arrays from source MIDI collections to update matrix parameter layouts.</p>', unsafe_allow_html=True)
    
    retrain_clicked = st.button("✦ Re-train model", use_container_width=True)
    if retrain_clicked:
        try:
            with st.spinner("Compiling structural graph layers... This deep training cycle runs across 100 complete validation steps."):
                train_script = os.path.join(base_dir, "train.py")
                res = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
                
            if res.returncode == 0:
                st.success("✦ Neural mapping success! The final weights matrix has been written to disk.")
                st.toast("Model retraining successful!", icon="🚀")
            else:
                st.error("Something went wrong during sequence compiling — make sure raw source sets are intact.")
        except Exception:
            st.error("Something went wrong — make sure your model is trained first.")

# PAGE 3: My Tracks
elif page_selection == "✦ My Tracks":
    st.markdown('<div class="title-accent">Cosmic Melody Archives</div>', unsafe_allow_html=True)
    st.markdown('<p class="secondary-text" style="font-size: 1.1rem; margin-top: 10px; margin-bottom: 30px;">Direct access points mapping to generated digital artifacts residing in your workspace path.</p>', unsafe_allow_html=True)
    
    try:
        mid_files = [f for f in os.listdir(base_dir) if f.endswith(".mid") and os.path.isfile(os.path.join(base_dir, f))]
        
        if mid_files:
            for m_file in sorted(mid_files):
                file_path = os.path.join(base_dir, m_file)
                try:
                    f_stats = os.stat(file_path)
                    f_size_kb = f_stats.st_size / 1024.0
                    f_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f_stats.st_ctime))
                except Exception:
                    f_size_kb = 0.0
                    f_date = "Unknown datestamp"
                    
                st.markdown(f"""
                <div class="cosmic-card" style="border-left: 4px solid #4DD9AC; margin-bottom: 16px; padding: 16px 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                        <div>
                            <p style="font-size: 1.15rem; font-weight: 700; color: #F5F0E8; margin: 0;">🎵 {m_file}</p>
                            <p style="font-size: 0.85rem; color: #8B7FA8; margin: 5px 0 0 0;">Timestamp: <b>{f_date}</b> &nbsp;|&nbsp; Artifact Weight: <b>{f_size_kb:.1f} KB</b></p>
                        </div>
                """, unsafe_allow_html=True)
                
                try:
                    with open(file_path, "rb") as f:
                        track_data = f.read()
                    st.download_button(
                        label=f"📥 Export File",
                        data=track_data,
                        file_name=m_file,
                        mime="audio/midi",
                        key=f"dl_{m_file}"
                    )
                except Exception:
                    st.error("Unable to resolve buffer stream.")
                    
                st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 70px 20px; background-color: #110e1f; border: 2px dashed #1e1535; border-radius: 12px; margin-top: 20px;">
                <p style="font-size: 4rem; margin: 0; color: #FF9A3C;">✨ ✦ ⭐</p>
                <p style="font-size: 1.4rem; font-weight: 700; color: #F5F0E8; margin-top: 18px; margin-bottom: 8px;">No cosmic compositions discovered yet</p>
                <p style="color: #8B7FA8; font-size: 1rem; max-width: 450px; margin: 0 auto;">Head over to the <b>Generate</b> tab to compose your very first AI masterpiece from stardust and imagination!</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.error("Something went wrong while indexing local MIDI archives.")
