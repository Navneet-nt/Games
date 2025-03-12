import streamlit as st
import time

class TicTacToeState:
    def __init__(self, board=None, player='X'):
        self.board = board if board else [' ' for _ in range(9)]
        self.player = player

    def get_empty_cells(self):
        return [i for i, cell in enumerate(self.board) if cell == ' ']

    def make_move(self, index):
        if self.board[index] != ' ':
            return None
        new_board = self.board.copy()
        new_board[index] = self.player
        next_player = 'O' if self.player == 'X' else 'X'
        return TicTacToeState(new_board, next_player)

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

def minimax(state, depth, is_maximizing, alpha, beta):
    terminal, winner, _ = state.is_terminal()
    if terminal:
        if winner == 'O':
            return 10 - depth
        elif winner == 'X':
            return depth - 10
        return 0
    
    if is_maximizing:
        best_score = -float('inf')
        for index in state.get_empty_cells():
            new_state = state.make_move(index)
            score = minimax(new_state, depth + 1, False, alpha, beta)
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = float('inf')
        for index in state.get_empty_cells():
            new_state = state.make_move(index)
            score = minimax(new_state, depth + 1, True, alpha, beta)
            best_score = min(best_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best_score

def best_move(state):
    best_score = -float('inf')
    move = None
    for index in state.get_empty_cells():
        new_state = state.make_move(index)
        score = minimax(new_state, 0, False, -float('inf'), float('inf'))
        if score > best_score:
            best_score = score
            move = index
    return move

def main():
    st.title("Tic-Tac-Toe ")
    
    if "game_state" not in st.session_state:
        st.session_state.game_state = TicTacToeState()
        st.session_state.current_player = 'X'
        st.session_state.winner = None
        st.session_state.winner_positions = None
    
    if "player_symbol" not in st.session_state:
        if st.button("Play as X (First)"):
            st.session_state.player_symbol = 'X'
            st.session_state.ai_symbol = 'O'
            st.rerun()
        if st.button("Play as O (Second)"):
            st.session_state.player_symbol = 'O'
            st.session_state.ai_symbol = 'X'
            st.session_state.current_player = 'X'
            st.rerun()
    else:
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                index = i * 3 + j
                button_label = st.session_state.game_state.board[index]
                disabled = button_label != ' ' or st.session_state.winner is not None or st.session_state.current_player != st.session_state.player_symbol
                if cols[j].button(button_label if button_label != ' ' else " ", key=f"cell_{index}", disabled=disabled):
                    st.session_state.game_state = st.session_state.game_state.make_move(index)
                    st.session_state.current_player = st.session_state.ai_symbol
                    st.rerun()
        
        terminal, winner, positions = st.session_state.game_state.is_terminal()
        if terminal and not st.session_state.winner:
            st.session_state.winner = winner if winner else "Draw"
            st.session_state.winner_positions = positions
            st.rerun()
        
        if st.session_state.current_player == st.session_state.ai_symbol and not st.session_state.winner:
            with st.spinner("AI is thinking..."):
                time.sleep(0.5)
                move = best_move(st.session_state.game_state)
                if move is not None:
                    st.session_state.game_state = st.session_state.game_state.make_move(move)
                    st.session_state.current_player = st.session_state.player_symbol
                    terminal, winner, positions = st.session_state.game_state.is_terminal()
                    if terminal:
                        st.session_state.winner = winner if winner else "Draw"
                        st.session_state.winner_positions = positions
                    st.rerun()
        
        if st.session_state.winner:
            if st.session_state.winner == st.session_state.player_symbol:
                st.success("You Win! 🏆")
            elif st.session_state.winner == st.session_state.ai_symbol:
                st.error("AI Wins! 🤖")
            else:
                st.warning("It's a Draw! 🤝")
            if st.button("Play Again"):
                st.session_state.game_state = TicTacToeState()
                st.session_state.current_player = 'X'
                st.session_state.winner = None
                st.session_state.winner_positions = None
                st.rerun()

if __name__ == "__main__":
    main()