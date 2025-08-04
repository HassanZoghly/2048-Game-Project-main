import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from game import Game2048
from agent import DQNAgent
import tensorflow as tf
import os
############################################# Set page configuration for a wide layout with a dark theme #############################################
st.set_page_config(
    page_title="2048 RL Agent",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)
############################################# Initialize session state for mode #############################################
if "game" not in st.session_state:
    st.session_state.game = Game2048()
if "ai_playing" not in st.session_state:
    st.session_state.ai_playing = False
if "game_result" not in st.session_state:
    st.session_state.game_result = None  # None, "win", or "lose"

# Initialize the agent without loading model
agent = DQNAgent()
############################################# Custom CSS for professional styling #############################################
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }
    
    /* Title styling */
    h1 {
        color: #00d4ff;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .intro-text {
        text-align: center;
        font-size: 1.1rem;
        color: #b0b0b0;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #16213e;
        padding: 1rem;
    }
    .sidebar-header {
        color: #00d4ff !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Game board styling */
    .game-board {
        display: grid;
        grid-template-columns: repeat(4, 100px);
        gap: 8px;
        background-color: #16213e;
        padding:50px;
        border-radius: 10px;
        box-shadow: 0 0 50px rgba(0, 0, 0, 5);
    }
    .game-tile {
        width: 100px;
        height: 100px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s ease-in-out;
        color: #1a1a2e;
    }
    .game-tile-appear {
        animation: tile-appear 0.3s ease-in-out;
    }
    @keyframes tile-appear {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    .tile-0 { background-color: #e0e0e0; }
    .tile-2 { background-color: #f5f5f5; }
    .tile-4 { background-color: #ece0c8; }
    .tile-8 { background-color: #f2b179; }
    .tile-16 { background-color: #f59563; }
    .tile-32 { background-color: #f67c5f; }
    .tile-64 { background-color: #f65e3b; }
    .tile-128 { background-color: #edcf72; }
    .tile-256 { background-color: #edcc61; }
    .tile-512 { background-color: #edc850; }
    .tile-1024 { background-color: #edc53f; }
    .tile-2048 { background-color: #edc22e; }
    
    /* Stats section */
    .stats-container {
        background-color: #16213e;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
        width: 100%;
        color: #e0e0e0;
    }
    
    /* Bubble animation for win */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); opacity: 0.5; }
        100% { transform: translateY(-800px) rotate(360deg); opacity: 0; }
    }
    .bubbles {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        z-index: 9999;
    }
    .bubble {
        position: absolute;
        background: rgba(255,255,255,0.6);
        border-radius: 50%;
        animation: float 5s infinite;
        opacity: 0;
    }
</style>
""", unsafe_allow_html=True)
############################################# Title and introduction #############################################
st.title("2048 Game - Reinforcement Learning")
st.markdown('<div class="intro-text">This application uses a Deep Q-Network (DQN) agent playing the 2048 game.</div>', unsafe_allow_html=True)
############################################# Initialize the game board display #############################################
def render_game_board(game):
    board = game.board
    board_html = '<div class="game-board">'
    for i in range(4):
        for j in range(4):
            tile_value = int(board[i, j])
            board_html += f'<div class="game-tile tile-{tile_value} game-tile-appear">{tile_value if tile_value != 0 else ""}</div>'
    board_html += '</div>'
    return board_html
############################################# Game section with board only #############################################
# Game section with board and message
game = st.session_state.game
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown(render_game_board(game), unsafe_allow_html=True)
with col2:
    if st.session_state.game_result == "lose":
        st.error("Game Over! Play Again! 🎮", icon="⚠️")
    elif st.session_state.game_result == "win":
        st.success("Congratulations! 🎉 You've Reached 2048!", icon="✅")
        st.markdown("""
        <div class="bubbles">
            <div class="bubble" style="left: 5%; width: 15px; height: 15px; animation-delay: 0.5s;"></div>
            <div class="bubble" style="left: 15%; width: 25px; height: 25px; animation-delay: 1.2s;"></div>
            <div class="bubble" style="left: 25%; width: 20px; height: 20px; animation-delay: 0.8s;"></div>
            <div class="bubble" style="left: 35%; width: 30px; height: 30px; animation-delay: 1.5s;"></div>
            <div class="bubble" style="left: 45%; width: 10px; height: 10px; animation-delay: 0.3s;"></div>
        </div>
        """, unsafe_allow_html=True)
with col3:
    pass
############################################# Game Controls and Stats in Sidebar #############################################
with st.sidebar:
    # Add CSS for psychedelic button effects
    st.markdown("""
    <style>
        /* Base button styling */
        .stButton>button {
            transition: all 0.5s ease !important;
            border: 2px solid transparent !important;
            position: relative;
            overflow: hidden;
            z-index: 1;
            color: white !important;
            font-weight: bold !important;
            text-shadow: 0 0 5px rgba(0,0,0,0.3) !important;
        }
        
        /* Rainbow border animation */
        .stButton>button:before {
            content: "";
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            z-index: -1;
            background: linear-gradient(
                45deg,
                #ff0000, #ff7300, #fffb00, 
                #48ff00, #00ffd5, #002bff, 
                #7a00ff, #ff00c8, #ff0000
            );
            background-size: 400%;
            border-radius: 8px;
            animation: rainbow-border 8s linear infinite;
        }
        
        @keyframes rainbow-border {
            0% { background-position: 0 0; }
            100% { background-position: 400% 0; }
        }
        
        /* Button inner color */
        .stButton>button:after {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: -1;
            background: #1a1a1a;
            border-radius: 6px;
            transition: all 0.3s ease;
        }
        
        /* Start AI button */
        .stButton>button:first-child:after {
            background: linear-gradient(135deg, #6e8efb, #a777e3) !important;
        }
        
        /* Reset Game button */
        .stButton>button:nth-child(2):after {
            background: linear-gradient(135deg, #f093fb, #f5576c) !important;
        }
        
        /* Hover effects */
        .stButton>button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3) !important;
        }
        
        .stButton>button:hover:after {
            opacity: 0.8;
        }
        
        /* Click effects */
        .stButton>button:active {
            transform: translateY(1px) scale(0.98) !important;
        }
        
        /* Pulsing glow effect */
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 5px rgba(255,255,255,0.5); }
            50% { box-shadow: 0 0 20px rgba(255,255,255,0.9); }
            100% { box-shadow: 0 0 5px rgba(255,255,255,0.5); }
        }
        
        .stButton>button:hover:before {
            animation: rainbow-border 3s linear infinite, pulse-glow 2s ease infinite;
        }
    </style>
    """, unsafe_allow_html=True)
    with st.sidebar:
        # Display score and steps in a styled container
        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
        st.markdown(f"**Score:** {game.score}")
        st.markdown(f"**Steps:** {game.moves}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Existing buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Start AI ✨", key="start_ai", use_container_width=True):
                st.session_state.ai_playing = True
        with btn_col2:
            if st.button("Reset Game ♻️", key="reset_game", use_container_width=True):
                st.session_state.game = Game2048()
                st.session_state.ai_playing = False
                st.session_state.game_result = None
                st.rerun()
############################################# AI playing logic #############################################
if st.session_state.ai_playing and not game.done:
    state = game.get_state()
    action = agent.get_action(state)
    _, _, done, info = game.step(action)
    game.score = info["score"]
    game.moves = info["moves"]
    
    if done:
        st.session_state.ai_playing = False
        max_tile = np.max(game.board)
        if max_tile >= 2048:
            st.session_state.game_result = "win"
        else:
            st.session_state.game_result = "lose"
    st.rerun()