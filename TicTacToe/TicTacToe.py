import streamlit as st
import heapq
import time

class TicTacToeState:
    def __init__(self, board=None, player='X', move=None):
        self.board = board if board else [' ' for _ in range(9)]
        self.player = player
        self.move = move
    
    def get_empty_cells(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']
    
    def make_move(self, index):
        if self.board[index] != ' ':
            return None
        new_board = self.board.copy()
        new_board[index] = self.player
        next_player = 'O' if self.player == 'X' else 'X'
        return TicTacToeState(new_board, next_player, index)
    
    def is_terminal(self):
        win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  
            [0, 4, 8], [2, 4, 6]              
        ]
        for positions in win_positions:
            if self.board[positions[0]] != ' ' and self.board[positions[0]] == self.board[positions[1]] == self.board[positions[2]]:
                return True, self.board[positions[0]], positions
        if ' ' not in self.board:
            return True, None, None
        return False, None, None
    
    def __lt__(self, other):
        """Defines how to compare two TicTacToeState objects in priority queue."""
        return id(self) < id(other)  # Use object ID for stable sorting

def heuristic(state, player):
    opponent = 'O' if player == 'X' else 'X'
    
    # Check terminal states
    terminal, winner, _ = state.is_terminal()
    if terminal:
        if winner == player:
            return -10
        elif winner == opponent:
            return 10
        else:
            return 0
    
    # Count threats (two in a row with an empty space)
    win_positions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    
    player_threats = 0
    opponent_threats = 0
    
    for positions in win_positions:
        line = [state.board[pos] for pos in positions]
        
        # Check for player threats
        if line.count(player) == 2 and line.count(' ') == 1:
            player_threats += 1
            
        # Check for opponent threats
        if line.count(opponent) == 2 and line.count(' ') == 1:
            opponent_threats += 1
    
    # Prioritize center and corners
    center_value = 1 if state.board[4] == player else 0
    corner_value = sum(1 for pos in [0, 2, 6, 8] if state.board[pos] == player) * 0.5
    
    # Return score based on threats and position quality
    return opponent_threats * 2 - player_threats * 1.5 - center_value - corner_value

def a_star_search(state, player):
    counter = 0
    open_set = [(heuristic(state, player), counter, state)]
    visited = set()
    
    start_time = time.time()
    best_move = None
    
    # If AI can win in one move, do it immediately
    for index in state.get_empty_cells():
        new_state = state.make_move(index)
        terminal, winner, _ = new_state.is_terminal()
        if terminal and winner == player:
            return index
    
    # If opponent can win in one move, block it
    opponent = 'X' if player == 'O' else 'O'
    for index in state.get_empty_cells():
        temp_board = state.board.copy()
        temp_board[index] = opponent
        temp_state = TicTacToeState(temp_board, player)
        terminal, winner, _ = temp_state.is_terminal()
        if terminal and winner == opponent:
            return index
    
    # Otherwise use A* to find best move
    while open_set and time.time() - start_time < 2:  # Add time limit to prevent long calculations
        _, _, current = heapq.heappop(open_set)
        
        if current.move is not None and best_move is None:
            best_move = current.move
            
        visited.add(tuple(current.board))
        
        for index in current.get_empty_cells():
            new_state = current.make_move(index)
            if tuple(new_state.board) in visited:
                continue
                
            terminal, winner, _ = new_state.is_terminal()
            if terminal and winner == player:
                return index
                
            counter += 1
            heapq.heappush(open_set, (heuristic(new_state, player), counter, new_state))
    
    # If no clear winning path, prefer center, then corners, then edges
    if 4 in state.get_empty_cells():
        return 4
    
    for corner in [0, 2, 6, 8]:
        if corner in state.get_empty_cells():
            return corner
            
    # Return best_move if found, otherwise first available move
    return best_move if best_move is not None else state.get_empty_cells()[0]

def get_winner_highlight(board, winner_positions=None):
    """Generates CSS for highlighting the winning positions"""
    highlights = {}
    
    if winner_positions:
        for pos in winner_positions:
            highlights[pos] = "background-color: #a8f0a8; color: #000000; font-weight: bold;"
    
    return highlights

def main():
    st.set_page_config(page_title="Tic-Tac-Toe A*", page_icon="🎮", layout="centered")
    
    st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
    }
    .button-row button {
        height: 80px;
        font-size: 24px !important;
    }
    .stButton>button {
        width: 100%;
        height: 80px;
        font-size: 28px !important;
    }
    .info-text {
        color: #555;
        text-align: center;
    }
    .winner-text {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 5px;
        margin: 15px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="big-font">Tic-Tac-Toe with A* AI</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if "game_state" not in st.session_state:
        st.session_state.game_state = TicTacToeState()
        st.session_state.current_player = 'X'
        st.session_state.winner = None
        st.session_state.winner_positions = None
        st.session_state.stats = {"ai_moves": 0, "player_moves": 0, "games_played": 0}

    # Player selection
    if "player_symbol" not in st.session_state:
        st.markdown('<p class="info-text">Choose your symbol:</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        if col1.button("Play as X (First)", key="choose_x"):
            st.session_state.player_symbol = 'X'
            st.session_state.ai_symbol = 'O'
            st.session_state.current_player = 'X'
            st.rerun()
        if col2.button("Play as O (Second)", key="choose_o"):
            st.session_state.player_symbol = 'O'
            st.session_state.ai_symbol = 'X'
            st.session_state.current_player = 'X'
            st.rerun()
    else:
        # Display current player
        if not st.session_state.winner:
            player_text = "Your turn" if st.session_state.current_player == st.session_state.player_symbol else "AI is thinking..."
            st.markdown(f'<p class="info-text">{player_text}</p>', unsafe_allow_html=True)
        
        # Get highlights for winning cells
        highlights = get_winner_highlight(st.session_state.game_state.board, st.session_state.winner_positions)
        
        # Game board
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                index = i * 3 + j
                current_value = st.session_state.game_state.board[index]
                
                # Determine button style based on cell content and if it's part of winning line
                if index in highlights:
                    button_style = highlights[index]
                elif current_value == 'X':
                    button_style = "color: #e53935; font-weight: bold;"
                elif current_value == 'O':
                    button_style = "color: #1e88e5; font-weight: bold;"
                else:
                    button_style = ""
                
                button_label = "  " if current_value == ' ' else current_value
                button_disabled = bool(current_value != ' ' or st.session_state.current_player != st.session_state.player_symbol or st.session_state.winner is not None)
                
                # Apply custom style to button
                if button_style:
                    button_html = f"""
                    <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child({j+1}) button {{
                        {button_style}
                    }}
                    </style>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)
                
                # Create the button
                if cols[j].button(button_label, key=f"cell_{index}", disabled=button_disabled):
                    # Player move
                    st.session_state.game_state = st.session_state.game_state.make_move(index)
                    st.session_state.current_player = st.session_state.ai_symbol
                    st.session_state.stats["player_moves"] += 1
                    st.rerun()
        
        # Check for game over
        terminal, winner, positions = st.session_state.game_state.is_terminal()
        if terminal and not st.session_state.winner:
            st.session_state.winner = winner if winner else "Draw"
            st.session_state.winner_positions = positions
            st.session_state.stats["games_played"] += 1
            st.rerun()

        
        # AI move
        if st.session_state.current_player == st.session_state.ai_symbol and not st.session_state.winner:
            with st.spinner("AI is thinking..."):
                time.sleep(0.5)  # Brief delay for better UX
                move = a_star_search(st.session_state.game_state, st.session_state.ai_symbol)
                if move is not None:
                    st.session_state.game_state = st.session_state.game_state.make_move(move)
                    st.session_state.current_player = st.session_state.player_symbol
                    st.session_state.stats["ai_moves"] += 1
                    
                    # Check if AI won
                    terminal, winner, positions = st.session_state.game_state.is_terminal()
                    if terminal:
                        st.session_state.winner = winner
                        st.session_state.winner_positions = positions
                        st.session_state.stats["games_played"] += 1
                    
                    st.rerun()
        
        # Display winner or draw
        if st.session_state.winner:
            if st.session_state.winner == st.session_state.player_symbol:
                st.markdown('<div class="winner-text" style="background-color: #a8f0a8;">You Win! 🏆</div>', unsafe_allow_html=True)
            elif st.session_state.winner == st.session_state.ai_symbol:
                st.markdown('<div class="winner-text" style="background-color: #f0a8a8;">AI Wins! 🤖</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="winner-text" style="background-color: #f0f0a8;">It\'s a Draw! 🤝</div>', unsafe_allow_html=True)
            
            # Play again button
            if st.button("Play Again"):
                st.session_state.game_state = TicTacToeState()
                st.session_state.current_player = 'X'
                st.session_state.winner = None
                st.session_state.winner_positions = None
                st.rerun()
        
        # Display stats in the sidebar
        st.sidebar.title("Game Statistics")
        st.sidebar.write(f"Games played: {st.session_state.stats['games_played']}")
        st.sidebar.write(f"Your moves: {st.session_state.stats['player_moves']}")
        st.sidebar.write(f"AI moves: {st.session_state.stats['ai_moves']}")
        
        # A* algorithm explanation
        with st.sidebar.expander("About A* Algorithm"):
            st.write("""
            The A* algorithm is a best-first search algorithm that finds the shortest path from a start node to a goal node.
            
            In this Tic-Tac-Toe implementation, A* uses a heuristic function that evaluates:
            - Winning and blocking opportunities
            - Strategic control of the center and corners
            - Potential threats (two in a row)
            
            The AI explores the game tree efficiently by prioritizing the most promising moves first.
            """)

if __name__ == "__main__":
    main()