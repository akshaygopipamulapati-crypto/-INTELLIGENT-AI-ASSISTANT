import os
import base64
import random
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neon AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HELPERS
# ============================================================

def html(markup: str):
    """Render HTML without markdown treating indented lines as code."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def svg_uri(svg: str) -> str:
    """Turn an SVG string into an inline data URI (no external files needed)."""
    encoded = base64.b64encode(svg.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


# ============================================================
# GENERATED IMAGES (INLINE ANIMATED SVGs)
# ============================================================

HERO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 320">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#061127"/>
      <stop offset="0.5" stop-color="#120a33"/>
      <stop offset="1" stop-color="#2a0a3a"/>
    </linearGradient>
    <radialGradient id="core" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#ffffff"/>
      <stop offset="0.25" stop-color="#00ffff"/>
      <stop offset="1" stop-color="#7c3aed" stop-opacity="0"/>
    </radialGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M40 0H0V40" fill="none" stroke="#00ffff" stroke-opacity="0.08"/>
    </pattern>
  </defs>
  <rect width="1200" height="320" fill="url(#bg)"/>
  <rect width="1200" height="320" fill="url(#grid)"/>

  <g transform="translate(600 160)" filter="url(#glow)">
    <circle r="140" fill="none" stroke="#00ffff" stroke-opacity="0.35" stroke-dasharray="6 10">
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="30s" repeatCount="indefinite"/>
    </circle>
    <circle r="100" fill="none" stroke="#ff00cc" stroke-opacity="0.5" stroke-dasharray="20 12">
      <animateTransform attributeName="transform" type="rotate" from="360" to="0" dur="18s" repeatCount="indefinite"/>
    </circle>
    <circle r="62" fill="none" stroke="#7c3aed" stroke-width="2">
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="10s" repeatCount="indefinite"/>
    </circle>
    <circle r="46" fill="url(#core)">
      <animate attributeName="r" values="40;54;40" dur="3s" repeatCount="indefinite"/>
    </circle>
    <g>
      <circle cx="140" cy="0" r="7" fill="#00ffff"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="360" dur="8s" repeatCount="indefinite"/>
    </g>
    <g>
      <circle cx="-100" cy="0" r="6" fill="#ff00cc"/>
      <animateTransform attributeName="transform" type="rotate" from="0" to="-360" dur="6s" repeatCount="indefinite"/>
    </g>
  </g>

  <g stroke="#00ffff" stroke-opacity="0.5" fill="none" filter="url(#glow)">
    <path d="M0 80 H220 L260 120 H420" stroke-dasharray="500" stroke-dashoffset="500">
      <animate attributeName="stroke-dashoffset" values="500;0;500" dur="6s" repeatCount="indefinite"/>
    </path>
    <path d="M1200 240 H980 L940 200 H780" stroke="#ff00cc" stroke-dasharray="500" stroke-dashoffset="500">
      <animate attributeName="stroke-dashoffset" values="500;0;500" dur="7s" repeatCount="indefinite"/>
    </path>
    <path d="M0 260 H140 L180 230 H360" stroke="#7c3aed" stroke-dasharray="400" stroke-dashoffset="400">
      <animate attributeName="stroke-dashoffset" values="400;0;400" dur="5s" repeatCount="indefinite"/>
    </path>
    <path d="M1200 70 H1050 L1010 100 H860" stroke-dasharray="400" stroke-dashoffset="400">
      <animate attributeName="stroke-dashoffset" values="400;0;400" dur="8s" repeatCount="indefinite"/>
    </path>
  </g>

  <g fill="#00ffff" filter="url(#glow)">
    <circle cx="260" cy="120" r="4"><animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/></circle>
    <circle cx="940" cy="200" r="4" fill="#ff00cc"><animate attributeName="opacity" values="0.2;1;0.2" dur="2.4s" repeatCount="indefinite"/></circle>
    <circle cx="180" cy="230" r="4" fill="#7c3aed"><animate attributeName="opacity" values="1;0.2;1" dur="1.8s" repeatCount="indefinite"/></circle>
    <circle cx="1010" cy="100" r="4"><animate attributeName="opacity" values="0.2;1;0.2" dur="2.2s" repeatCount="indefinite"/></circle>
  </g>
</svg>
"""

ICON_SEARCH = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#00ffff"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
  <circle cx="42" cy="42" r="26" fill="none" stroke="url(#g)" stroke-width="7"/>
  <line x1="62" y1="62" x2="88" y2="88" stroke="url(#g)" stroke-width="9" stroke-linecap="round"/>
  <path d="M28 42 H56 M42 28 V56" stroke="#00ffff" stroke-opacity="0.5" stroke-width="2"/>
  <circle cx="42" cy="42" r="26" fill="none" stroke="#fff" stroke-opacity="0.6" stroke-width="2" stroke-dasharray="10 150">
    <animateTransform attributeName="transform" type="rotate" from="0 42 42" to="360 42 42" dur="3s" repeatCount="indefinite"/>
  </circle>
</svg>
"""

ICON_CALC = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ff00cc"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
  <rect x="20" y="10" width="60" height="80" rx="10" fill="none" stroke="url(#g)" stroke-width="6"/>
  <rect x="28" y="18" width="44" height="18" rx="4" fill="#00ffff" fill-opacity="0.25">
    <animate attributeName="fill-opacity" values="0.15;0.5;0.15" dur="2s" repeatCount="indefinite"/>
  </rect>
  <g fill="url(#g)">
    <circle cx="35" cy="52" r="5"/><circle cx="50" cy="52" r="5"/><circle cx="65" cy="52" r="5"/>
    <circle cx="35" cy="68" r="5"/><circle cx="50" cy="68" r="5"/><circle cx="65" cy="68" r="5"/>
    <circle cx="35" cy="82" r="3"/><circle cx="50" cy="82" r="3"/><circle cx="65" cy="82" r="3"/>
  </g>
</svg>
"""

ICON_STOCK = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs><linearGradient id="g" x1="0" y1="1" x2="1" y2="0">
    <stop offset="0" stop-color="#7c3aed"/><stop offset="1" stop-color="#22ff88"/></linearGradient></defs>
  <path d="M10 88 H92 M10 88 V12" stroke="#94a3b8" stroke-opacity="0.5" stroke-width="3" fill="none"/>
  <g fill="url(#g)">
    <rect x="20" y="58" width="12" height="30" rx="2"/>
    <rect x="38" y="44" width="12" height="44" rx="2"/>
    <rect x="56" y="50" width="12" height="38" rx="2"/>
    <rect x="74" y="26" width="12" height="62" rx="2"/>
  </g>
  <polyline points="14,62 34,40 52,48 80,14" fill="none" stroke="#00ffff" stroke-width="4"
            stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="140" stroke-dashoffset="140">
    <animate attributeName="stroke-dashoffset" values="140;0;0;140" keyTimes="0;0.5;0.85;1" dur="4s" repeatCount="indefinite"/>
  </polyline>
</svg>
"""

ICON_WEATHER = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#00ffff"/><stop offset="1" stop-color="#ff00cc"/></linearGradient></defs>
  <g>
    <circle cx="66" cy="34" r="14" fill="#ffd54a"/>
    <g stroke="#ffd54a" stroke-width="4" stroke-linecap="round">
      <line x1="66" y1="8" x2="66" y2="14"/><line x1="66" y1="54" x2="66" y2="60"/>
      <line x1="40" y1="34" x2="46" y2="34"/><line x1="86" y1="34" x2="92" y2="34"/>
    </g>
    <animateTransform attributeName="transform" type="rotate" from="0 66 34" to="360 66 34" dur="14s" repeatCount="indefinite"/>
  </g>
  <path d="M28 76 a14 14 0 0 1 2 -28 a20 20 0 0 1 38 6 a12 12 0 0 1 -2 22 z"
        fill="#0b1230" stroke="url(#g)" stroke-width="4">
    <animateTransform attributeName="transform" type="translate" values="0 0; 4 0; 0 0" dur="5s" repeatCount="indefinite"/>
  </path>
  <g stroke="#00ffff" stroke-width="3" stroke-linecap="round">
    <line x1="36" y1="84" x2="33" y2="92"><animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/></line>
    <line x1="50" y1="84" x2="47" y2="92"><animate attributeName="opacity" values="0;1;0" dur="1s" repeatCount="indefinite"/></line>
    <line x1="64" y1="84" x2="61" y2="92"><animate attributeName="opacity" values="1;0;1" dur="1.3s" repeatCount="indefinite"/></line>
  </g>
</svg>
"""

LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#00ffff"/><stop offset="0.5" stop-color="#7c3aed"/><stop offset="1" stop-color="#ff00cc"/>
    </linearGradient>
    <filter id="f"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <g filter="url(#f)">
    <polygon points="100,12 178,56 178,144 100,188 22,144 22,56" fill="#070c1f" stroke="url(#g)" stroke-width="6">
      <animate attributeName="stroke-opacity" values="1;0.5;1" dur="3s" repeatCount="indefinite"/>
    </polygon>
    <rect x="62" y="66" width="76" height="60" rx="14" fill="none" stroke="url(#g)" stroke-width="6"/>
    <circle cx="84" cy="94" r="8" fill="#00ffff"><animate attributeName="r" values="8;5;8" dur="2.5s" repeatCount="indefinite"/></circle>
    <circle cx="116" cy="94" r="8" fill="#ff00cc"><animate attributeName="r" values="5;8;5" dur="2.5s" repeatCount="indefinite"/></circle>
    <line x1="84" y1="114" x2="116" y2="114" stroke="url(#g)" stroke-width="5" stroke-linecap="round"/>
    <line x1="100" y1="66" x2="100" y2="46" stroke="url(#g)" stroke-width="5"/>
    <circle cx="100" cy="42" r="6" fill="#00ffff"><animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/></circle>
  </g>
</svg>
"""

HERO_IMG = svg_uri(HERO_SVG)
LOGO_IMG = svg_uri(LOGO_SVG)
ICONS = {
    "search": svg_uri(ICON_SEARCH),
    "calc": svg_uri(ICON_CALC),
    "stock": svg_uri(ICON_STOCK),
    "weather": svg_uri(ICON_WEATHER),
}


# ============================================================
# GAMING / NEON CSS  (with extra effects)
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&display=swap');

/* ---------- animatable custom property for rotating borders ---------- */
@property --angle {
    syntax: "<angle>";
    initial-value: 0deg;
    inherits: false;
}

/* =========================================================
   GLOBAL BACKGROUND
========================================================= */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(0,255,255,0.16), transparent 25%),
        radial-gradient(circle at 90% 15%, rgba(255,0,180,0.16), transparent 25%),
        radial-gradient(circle at 50% 90%, rgba(110,0,255,0.18), transparent 30%),
        linear-gradient(135deg, #030712 0%, #080d20 45%, #0c1029 100%);
    background-size: 140% 140%, 140% 140%, 140% 140%, 100% 100%;
    animation: bgShift 18s ease-in-out infinite alternate;
    color: white;
}

@keyframes bgShift {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0 0; }
    100% { background-position: 30% 20%, 70% 30%, 40% 70%, 0 0; }
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    position: relative;
    z-index: 2;
}

/* =========================================================
   EFFECT: MOVING CYBER GRID
========================================================= */

.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,255,255,0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: gridMove 12s linear infinite;
    pointer-events: none;
    z-index: 0;
    mask-image: linear-gradient(to bottom, transparent, black 30%, black 70%, transparent);
    -webkit-mask-image: linear-gradient(to bottom, transparent, black 30%, black 70%, transparent);
}

@keyframes gridMove {
    from { background-position: 0 0, 0 0; }
    to   { background-position: 0 50px, 50px 0; }
}

/* =========================================================
   EFFECT: FLOATING ORB
========================================================= */

.stApp::before {
    content: "";
    position: fixed;
    width: 450px;
    height: 450px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,255,255,0.12), transparent 70%);
    filter: blur(40px);
    animation: floatingGlow 10s infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes floatingGlow {
    0%   { transform: translate(-100px, -50px); }
    50%  { transform: translate(500px, 150px); }
    100% { transform: translate(900px, -100px); }
}

/* =========================================================
   EFFECT: SCANLINE SWEEP
========================================================= */

.scanline {
    position: fixed;
    left: 0;
    width: 100%;
    height: 120px;
    background: linear-gradient(to bottom, transparent, rgba(0,255,255,0.05), transparent);
    animation: scan 7s linear infinite;
    pointer-events: none;
    z-index: 1;
}

@keyframes scan {
    from { top: -120px; }
    to   { top: 100%; }
}

/* =========================================================
   EFFECT: RISING PARTICLES
========================================================= */

.particles {
    position: fixed;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 1;
}

.particles span {
    position: absolute;
    bottom: -20px;
    border-radius: 50%;
    background: #00ffff;
    box-shadow: 0 0 12px #00ffff, 0 0 24px rgba(0,255,255,0.6);
    opacity: 0;
    animation: rise linear infinite;
}

.particles span:nth-child(3n)   { background: #ff00cc; box-shadow: 0 0 12px #ff00cc; }
.particles span:nth-child(3n+1) { background: #7c3aed; box-shadow: 0 0 12px #7c3aed; }

@keyframes rise {
    0%   { transform: translateY(0) translateX(0); opacity: 0; }
    10%  { opacity: 0.9; }
    90%  { opacity: 0.6; }
    100% { transform: translateY(-110vh) translateX(40px); opacity: 0; }
}

/* =========================================================
   HERO BANNER (generated image)
========================================================= */

.hero-wrap {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 10px;
    padding: 2px;
    background: conic-gradient(from var(--angle), #00ffff, #7c3aed, #ff00cc, #00ffff);
    animation: spinBorder 6s linear infinite;
    box-shadow: 0 0 40px rgba(0,255,255,0.18), 0 0 80px rgba(255,0,204,0.12);
}

@keyframes spinBorder {
    to { --angle: 360deg; }
}

.hero-inner {
    position: relative;
    border-radius: 22px;
    overflow: hidden;
    background: #050816;
}

.hero-inner img {
    display: block;
    width: 100%;
    height: auto;
    animation: heroBreath 8s ease-in-out infinite alternate;
}

@keyframes heroBreath {
    from { transform: scale(1);    filter: saturate(1)   brightness(0.95); }
    to   { transform: scale(1.04); filter: saturate(1.3) brightness(1.1); }
}

.hero-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: radial-gradient(circle, rgba(3,7,18,0.15), rgba(3,7,18,0.65));
}

/* =========================================================
   ONLINE STATUS
========================================================= */

.online-status {
    text-align: center;
    color: #22ff88;
    font-size: 14px;
    font-weight: 800;
    letter-spacing: 2px;
    text-shadow: 0 0 10px #22ff88;
    margin-bottom: 8px;
    animation: pulseStatus 2s infinite;
}

@keyframes pulseStatus {
    0%   { opacity: 0.6; }
    50%  { opacity: 1; }
    100% { opacity: 0.6; }
}

/* =========================================================
   TITLE  (rainbow + glitch)
========================================================= */

.game-title {
    position: relative;
    text-align: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 48px;
    font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(90deg, #00ffff, #7c3aed, #ff00cc, #00ffff);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: rainbowText 5s linear infinite, glitch 6s steps(1) infinite;
    filter: drop-shadow(0 0 18px rgba(0,255,255,0.35));
    margin-bottom: 5px;
}

@keyframes rainbowText {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes glitch {
    0%, 92%, 100% { transform: translate(0); }
    93% { transform: translate(-3px, 1px) skewX(4deg); }
    94% { transform: translate(3px, -1px) skewX(-4deg); }
    95% { transform: translate(-2px, 0); }
    96% { transform: translate(0); }
}

/* =========================================================
   SUBTITLE (typewriter)
========================================================= */

.game-subtitle {
    text-align: center;
    margin: 0 auto 25px auto;
    width: fit-content;
    max-width: 100%;
    color: #a5b4fc;
    font-size: 17px;
    white-space: nowrap;
    overflow: hidden;
    border-right: 3px solid #00ffff;
    animation: typing 3.5s steps(60, end) 0.4s both, caret 0.8s step-end infinite;
}

@keyframes typing {
    from { width: 0; }
    to   { width: 100%; }
}

@keyframes caret {
    50% { border-color: transparent; }
}

/* =========================================================
   TICKER
========================================================= */

.ticker {
    overflow: hidden;
    white-space: nowrap;
    border-top: 1px solid rgba(0,255,255,0.2);
    border-bottom: 1px solid rgba(0,255,255,0.2);
    background: rgba(0,255,255,0.04);
    padding: 8px 0;
    margin: 10px 0 20px 0;
}

.ticker-track {
    display: inline-block;
    padding-left: 100%;
    color: #7dd3fc;
    font-size: 13px;
    letter-spacing: 2px;
    animation: marquee 28s linear infinite;
}

.ticker:hover .ticker-track { animation-play-state: paused; }

@keyframes marquee {
    from { transform: translateX(0); }
    to   { transform: translateX(-100%); }
}

/* =========================================================
   SECTION HEADINGS
========================================================= */

.section-heading {
    font-family: 'Orbitron', sans-serif;
    font-size: 22px;
    font-weight: 700;
    color: white;
    margin-top: 30px;
    margin-bottom: 15px;
    padding-left: 14px;
    border-left: 4px solid #00ffff;
    text-shadow: 0 0 10px rgba(0,255,255,0.35);
    animation: headingSlide 0.9s ease both;
}

@keyframes headingSlide {
    from { opacity: 0; transform: translateX(-30px); }
    to   { opacity: 1; transform: translateX(0); }
}

/* =========================================================
   TOOL CARDS  (rotating neon border + image + shine)
========================================================= */

.game-card {
    position: relative;
    min-height: 210px;
    padding: 22px 15px;
    border-radius: 20px;
    overflow: hidden;
    background: linear-gradient(145deg, rgba(17,24,50,0.95), rgba(7,12,30,0.95));
    margin-bottom: 15px;
    transition: all 0.3s ease;
    animation: cardIn 0.8s ease both;
    isolation: isolate;
}

/* rotating border */
.game-card::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 2px;
    border-radius: 20px;
    background: conic-gradient(from var(--angle), transparent 0%, #00ffff 15%, transparent 30%, #ff00cc 60%, transparent 80%);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    animation: spinBorder 5s linear infinite;
    z-index: 1;
    pointer-events: none;
}

/* light shine sweep */
.game-card::after {
    content: "";
    position: absolute;
    top: 0;
    left: -120%;
    width: 60%;
    height: 100%;
    background: linear-gradient(105deg, transparent, rgba(255,255,255,0.14), transparent);
    transform: skewX(-20deg);
    transition: left 0.7s ease;
    z-index: 2;
    pointer-events: none;
}

.game-card:hover::after { left: 140%; }

.game-card:hover {
    transform: translateY(-10px) scale(1.04) rotateX(3deg);
    box-shadow: 0 0 25px rgba(0,255,255,0.40), 0 0 60px rgba(120,0,255,0.25);
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(40px) scale(0.92); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

.c-delay-1 { animation-delay: 0.10s; }
.c-delay-2 { animation-delay: 0.25s; }
.c-delay-3 { animation-delay: 0.40s; }
.c-delay-4 { animation-delay: 0.55s; }

.tool-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 10px;
    animation: floaty 4s ease-in-out infinite;
    filter: drop-shadow(0 0 12px rgba(0,255,255,0.6));
}

.tool-icon img { width: 78px; height: 78px; }

@keyframes floaty {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}

.tool-name {
    text-align: center;
    color: white;
    font-family: 'Orbitron', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 1px;
}

.tool-description {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 7px;
}

/* =========================================================
   STAT CARDS
========================================================= */

.stat-card {
    position: relative;
    text-align: center;
    padding: 20px;
    border-radius: 18px;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(0,255,255,0.08), rgba(120,0,255,0.12));
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 0 20px rgba(120,0,255,0.08);
    transition: all 0.3s ease;
    animation: cardIn 0.8s ease both;
}

.stat-card::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: 0;
    height: 3px;
    width: 100%;
    background: linear-gradient(90deg, #00ffff, #ff00cc);
    transform-origin: left;
    animation: barGrow 3s ease-in-out infinite alternate;
}

@keyframes barGrow {
    from { transform: scaleX(0.15); }
    to   { transform: scaleX(1); }
}

.stat-card:hover {
    transform: translateY(-5px);
    border-color: rgba(0,255,255,0.5);
    box-shadow: 0 0 25px rgba(0,255,255,0.20);
}

.stat-number {
    font-family: 'Orbitron', sans-serif;
    font-size: 32px;
    font-weight: 900;
    color: #00ffff;
    text-shadow: 0 0 12px rgba(0,255,255,0.7);
    animation: neonFlicker 4s infinite;
}

@keyframes neonFlicker {
    0%, 18%, 22%, 62%, 64%, 100% { opacity: 1; text-shadow: 0 0 12px rgba(0,255,255,0.8), 0 0 30px rgba(0,255,255,0.5); }
    20%, 63% { opacity: 0.55; text-shadow: none; }
}

.stat-label {
    color: #94a3b8;
    font-size: 12px;
    letter-spacing: 1px;
}

/* =========================================================
   BUTTONS  (shimmer + ripple-ish glow)
========================================================= */

.stButton > button {
    position: relative;
    overflow: hidden;
    width: 100%;
    min-height: 48px;
    border-radius: 14px;
    border: 1px solid rgba(0,255,255,0.35);
    background: linear-gradient(135deg, rgba(0,255,255,0.10), rgba(120,0,255,0.18));
    color: white;
    font-weight: 700;
    transition: all 0.25s ease;
    box-shadow: 0 0 10px rgba(0,255,255,0.05);
}

.stButton > button::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(100deg, transparent, rgba(255,255,255,0.22), transparent);
    transition: left 0.6s ease;
}

.stButton > button:hover::before { left: 140%; }

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0,255,255,0.20), rgba(255,0,200,0.20));
    box-shadow: 0 0 20px rgba(0,255,255,0.35);
    color: white;
}

.stButton > button:active {
    transform: scale(0.96);
    box-shadow: 0 0 35px rgba(255,0,204,0.6);
}

/* =========================================================
   CHAT MESSAGES
========================================================= */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid rgba(0,255,255,0.12);
    background: rgba(10,15,35,0.78);
    backdrop-filter: blur(6px);
    box-shadow: 0 0 15px rgba(0,255,255,0.04);
    margin-bottom: 10px;
    transition: all 0.2s ease;
    animation: msgIn 0.5s cubic-bezier(.2,.9,.3,1.2) both;
}

@keyframes msgIn {
    from { opacity: 0; transform: translateY(20px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(0,255,255,0.30);
    box-shadow: 0 0 20px rgba(0,255,255,0.08);
}

/* =========================================================
   CHAT INPUT  (glow on focus)
========================================================= */

[data-testid="stChatInput"] {
    border-radius: 20px;
    box-shadow: 0 0 20px rgba(0,255,255,0.10);
    transition: box-shadow 0.3s ease;
}

[data-testid="stChatInput"]:focus-within {
    box-shadow: 0 0 0 1px #00ffff, 0 0 30px rgba(0,255,255,0.45), 0 0 60px rgba(255,0,204,0.20);
}

/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030712, #080d20, #100b25);
    border-right: 1px solid rgba(0,255,255,0.15);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #00ffff;
    text-shadow: 0 0 10px rgba(0,255,255,0.5);
}

.sidebar-logo {
    display: flex;
    justify-content: center;
    margin: 4px 0 8px 0;
}

.sidebar-logo img {
    width: 120px;
    animation: floaty 4s ease-in-out infinite;
    filter: drop-shadow(0 0 18px rgba(0,255,255,0.55));
}

/* =========================================================
   FOOTER
========================================================= */

.game-footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
    margin-top: 40px;
    font-size: 13px;
}

.game-footer b {
    background: linear-gradient(90deg, #00ffff, #ff00cc);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* =========================================================
   EXPANDER / SPINNER
========================================================= */

[data-testid="stExpander"] {
    border: 1px solid rgba(0,255,255,0.15);
    border-radius: 15px;
    background: rgba(10,15,35,0.6);
}

[data-testid="stSpinner"] {
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
    animation: pulseStatus 1.2s infinite;
}

/* =========================================================
   SCROLLBAR
========================================================= */

::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #050816; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(#00ffff, #7c3aed);
    border-radius: 10px;
}

/* =========================================================
   ACCESSIBILITY
========================================================= */

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKGROUND EFFECT LAYERS (particles + scanline)
# ============================================================

random.seed(7)

particle_html = ""

for _ in range(28):

    size = random.randint(3, 8)
    left = random.randint(0, 100)
    duration = random.randint(9, 22)
    delay = random.randint(0, 14)

    particle_html += (
        f'<span style="left:{left}%;width:{size}px;height:{size}px;'
        f'animation-duration:{duration}s;animation-delay:{delay}s;"></span>'
    )

html(f'<div class="particles">{particle_html}</div><div class="scanline"></div>')


# ============================================================
# API KEY CHECK
# ============================================================

missing_keys = []

if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if not TAVILY_API_KEY:
    missing_keys.append("TAVILY_API_KEY")

if not OPENWEATHER_API_KEY:
    missing_keys.append("OPENWEATHER_API_KEY")

if not ALPHA_API_KEY:
    missing_keys.append("ALPHA_API_KEY")


# ============================================================
# TOOLS
# ============================================================

@tool
def web_search(query: str) -> str:
    """
    Search the internet using Tavily.
    """

    if not TAVILY_API_KEY:
        return "Tavily API key is missing."

    try:

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "advanced",
                "max_results": 5
            },
            timeout=20
        )

        data = response.json()

        results = data.get("results", [])

        if not results:
            return "No search results found."

        output = []

        for item in results:

            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")

            output.append(
                f"Title: {title}\n"
                f"Information: {content}\n"
                f"URL: {url}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return f"Web search error: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    """

    try:

        allowed = "0123456789+-*/().% "

        if not all(char in allowed for char in expression):

            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})

        return str(result)

    except Exception as e:

        return f"Calculation error: {str(e)}"


@tool
def stock_price(symbol: str) -> str:
    """
    Get current stock information from Alpha Vantage.
    """

    if not ALPHA_API_KEY:

        return "Alpha Vantage API key is missing."

    try:

        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol.upper()}"
            f"&apikey={ALPHA_API_KEY}"
        )

        response = requests.get(url, timeout=15)

        data = response.json()

        quote = data.get("Global Quote", {})

        if not quote:

            return (
                "No stock data found. "
                "Check the stock symbol."
            )

        price = quote.get("05. price", "N/A")
        change = quote.get("09. change", "N/A")
        percent = quote.get("10. change percent", "N/A")

        return (
            f"Stock: {symbol.upper()}\n"
            f"Price: ${price}\n"
            f"Change: {change}\n"
            f"Change Percent: {percent}"
        )

    except Exception as e:

        return f"Stock API error: {str(e)}"


@tool
def weather(city: str) -> str:
    """
    Get current weather information for a city.
    """

    if not OPENWEATHER_API_KEY:

        return "OpenWeather API key is missing."

    try:

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}"
            f"&appid={OPENWEATHER_API_KEY}"
            f"&units=metric"
        )

        response = requests.get(url, timeout=15)

        data = response.json()

        if response.status_code != 200:

            return (
                f"Weather error: "
                f"{data.get('message', 'Unknown error')}"
            )

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"City: {city}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels Like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Condition: {description}"
        )

    except Exception as e:

        return f"Weather API error: {str(e)}"


# ============================================================
# AI MODEL
# ============================================================

if GROQ_API_KEY:

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2,
        api_key=GROQ_API_KEY
    )

else:

    llm = None


# ============================================================
# TOOLS LIST
# ============================================================

tools = [
    web_search,
    calculator,
    stock_price,
    weather
]


if llm:

    llm_with_tools = llm.bind_tools(tools)

else:

    llm_with_tools = None


# ============================================================
# LANGGRAPH STATE
# ============================================================

class State(TypedDict):

    messages: Annotated[list, add_messages]


# ============================================================
# CHATBOT NODE
# ============================================================

SYSTEM_PROMPT = """
You are an intelligent multi-tool AI assistant.

You can use these tools:

1. web_search
   - Use for current information and internet searches.

2. calculator
   - Use for mathematical calculations.

3. stock_price
   - Use for stock prices and market information.

4. weather
   - Use for weather information.

Use tools whenever they are appropriate.

Give clear, useful and easy-to-understand answers.

For tool results, explain the result naturally instead of
simply copying raw API output.
"""


def chatbot(state: State):

    if not llm_with_tools:

        return {
            "messages": [
                AIMessage(
                    content=(
                        "⚠️ GROQ_API_KEY is missing. "
                        "Please add it to your .env file."
                    )
                )
            ]
        }

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


# ============================================================
# BUILD LANGGRAPH
# ============================================================

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")

# tools_condition routes to "tools" or END by itself,
# so no extra chatbot -> END edge is needed.
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END}
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "demo_question" not in st.session_state:

    st.session_state.demo_question = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    html(
        f'<div class="sidebar-logo"><img src="{LOGO_IMG}" alt="logo"></div>'
    )

    st.markdown("## 🎮 AI CONTROL PANEL")

    st.markdown("---")

    st.markdown("### ⚡ SYSTEM")

    if GROQ_API_KEY:
        st.success("🟢 Groq AI Connected")
    else:
        st.error("🔴 Groq API Missing")

    if TAVILY_API_KEY:
        st.success("🟢 Web Search Ready")
    else:
        st.warning("🟡 Web Search Disabled")

    if OPENWEATHER_API_KEY:
        st.success("🟢 Weather Ready")
    else:
        st.warning("🟡 Weather Disabled")

    if ALPHA_API_KEY:
        st.success("🟢 Stock API Ready")
    else:
        st.warning("🟡 Stock API Disabled")

    st.markdown("---")

    st.markdown("### 🧠 AI ENGINE")

    st.info("LangGraph + ChatGroq")

    st.markdown("### 🛠️ TOOLS")

    st.write("🔎 Web Search")
    st.write("🧮 Calculator")
    st.write("📈 Stock Data")
    st.write("🌤️ Weather")

    st.markdown("---")

    if st.button("🗑️ CLEAR CHAT", use_container_width=True):

        st.session_state.messages = []

        st.toast("Chat cleared", icon="🗑️")

        st.rerun()


# ============================================================
# MAIN HEADER  (hero image + title)
# ============================================================

html(
    f"""
    <div class="hero-wrap">
        <div class="hero-inner">
            <img src="{HERO_IMG}" alt="AI core banner">
            <div class="hero-overlay">
                <div class="online-status">● AI SYSTEM ONLINE</div>
                <div class="game-title">🤖 INTELLIGENT AI ASSISTANT</div>
            </div>
        </div>
    </div>
    """
)

html(
    '<div class="game-subtitle">'
    '⚡ Multi-Tool AI • Real-Time Intelligence • Smart Automation'
    '</div>'
)

html(
    """
    <div class="ticker">
        <div class="ticker-track">
            🔎 WEB SEARCH ONLINE &nbsp;&nbsp;•&nbsp;&nbsp;
            🧮 CALCULATOR READY &nbsp;&nbsp;•&nbsp;&nbsp;
            📈 MARKET DATA STREAMING &nbsp;&nbsp;•&nbsp;&nbsp;
            🌤️ WEATHER SATELLITES LOCKED &nbsp;&nbsp;•&nbsp;&nbsp;
            ⚡ LANGGRAPH ENGINE ACTIVE &nbsp;&nbsp;•&nbsp;&nbsp;
            🧠 GROQ INFERENCE READY
        </div>
    </div>
    """
)


# ============================================================
# STATISTICS
# ============================================================

s1, s2, s3, s4 = st.columns(4)

stats = [
    ("4+", "AI TOOLS"),
    ("AI", "INTELLIGENCE"),
    ("LIVE", "DATA ACCESS"),
    ("∞", "QUESTIONS"),
]

for col, (number, label) in zip((s1, s2, s3, s4), stats):

    with col:

        html(
            f"""
            <div class="stat-card">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """
        )


# ============================================================
# AI POWER UPS
# ============================================================

html('<div class="section-heading">🎮 AI POWER-UPS</div>')

c1, c2, c3, c4 = st.columns(4)

cards = [
    (c1, "search", "WEB SEARCH", "Real-time internet information", "c-delay-1"),
    (c2, "calc", "CALCULATOR", "Mathematical computation", "c-delay-2"),
    (c3, "stock", "STOCK DATA", "Market information", "c-delay-3"),
    (c4, "weather", "WEATHER", "Live weather information", "c-delay-4"),
]

for col, key, name, desc, delay in cards:

    with col:

        html(
            f"""
            <div class="game-card {delay}">
                <div class="tool-icon"><img src="{ICONS[key]}" alt="{name}"></div>
                <div class="tool-name">{name}</div>
                <div class="tool-description">{desc}</div>
            </div>
            """
        )


# ============================================================
# DEMO MISSIONS
# ============================================================

html('<div class="section-heading">🎯 DEMO MISSIONS</div>')

d1, d2, d3, d4 = st.columns(4)

with d1:

    if st.button("🔎 SEARCH AI NEWS", use_container_width=True):

        st.session_state.demo_question = (
            "Search the web and explain the latest developments in Artificial Intelligence."
        )

with d2:

    if st.button("🧮 SOLVE CALCULATION", use_container_width=True):

        st.session_state.demo_question = "Calculate 125 * 48 + 250."

with d3:

    if st.button("🌤️ CHECK WEATHER", use_container_width=True):

        st.session_state.demo_question = (
            "What is the current weather in Hyderabad?"
        )

with d4:

    if st.button("📈 CHECK STOCK", use_container_width=True):

        st.session_state.demo_question = (
            "What is the current stock price of AAPL?"
        )


# ============================================================
# CHAT HISTORY
# ============================================================

html('<div class="section-heading">💬 AI COMMAND CENTER</div>')


for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message("user", avatar="🎮"):

            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        if message.content:

            with st.chat_message("assistant", avatar="🤖"):

                st.markdown(message.content)


# ============================================================
# DEMO QUESTION
# ============================================================

if st.session_state.demo_question:

    question = st.session_state.demo_question

    st.session_state.demo_question = None

else:

    question = st.chat_input("🎮 Enter your command...")


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if question:

    st.session_state.messages.append(HumanMessage(content=question))

    with st.chat_message("user", avatar="🎮"):

        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("⚡ AI PROCESSING..."):

            try:

                result = graph.invoke(
                    {"messages": st.session_state.messages}
                )

                response = result["messages"][-1]

                if isinstance(response, AIMessage):

                    answer = response.content

                else:

                    answer = str(response)

                st.markdown(answer)

                st.session_state.messages.append(
                    AIMessage(content=answer)
                )

                st.toast("Response received", icon="✅")

            except Exception as e:

                st.error(
                    f"⚠️ **AI Error**\n\n`{str(e)}`"
                )


# ============================================================
# FOOTER
# ============================================================

html(
    """
    <div class="game-footer">
        🤖 <b>Intelligent AI Assistant</b>
        &nbsp; • &nbsp;
        Powered by LangGraph + ChatGroq
        &nbsp; • &nbsp;
        ⚡ Multi-Tool AI System
        <br><br>
        🔎 Search &nbsp;|&nbsp;
        🧮 Calculate &nbsp;|&nbsp;
        📈 Analyze &nbsp;|&nbsp;
        🌤️ Monitor
    </div>
    """
)