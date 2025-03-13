import copy
from collections import deque
# Constants
HUMAN = 'X'
BOT = 'O'
EMPTY = ' '


# Initialize board
def create_board():
    return [[EMPTY for _ in range(3)] for _ in range(3)]


def print_board(board):
    for row in board:
        print('|'.join(row))
        print('-' * 5)


def is_winner(board, player):
    # Check rows, columns, diagonals
    for i in range(3):
        if all(cell == player for cell in board[i]):
            return True
        if all(row[i] == player for row in board):
            return True
    if all(board[i][i] == player for i in range(3)):
        return True
    if all(board[i][2 - i] == player for i in range(3)):
        return True
    return False


def is_full(board):
    return all(cell != EMPTY for row in board for cell in row)


def game_over(board):
    return is_winner(board, HUMAN) or is_winner(board, BOT) or is_full(board)


# Heuristic function
def evaluate(board, depth):
    if is_winner(board, BOT):
        return 1 / depth
    elif is_winner(board, HUMAN):
        return -1 * depth
    else:
        return 0


# DFS algorithm with heuristic
def dfs(board, is_bot_turn, depth=1):
    if game_over(board):
        return evaluate(board, depth), None

    best_move = None
    best_score = float('-inf') if is_bot_turn else float('inf')

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                board[i][j] = BOT if is_bot_turn else HUMAN
                score, _ = dfs(board, not is_bot_turn, depth=depth + 1)
                board[i][j] = EMPTY

                if is_bot_turn:
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = (i, j)
    return best_score, best_move


def bfs(board, is_bot_turn):
    queue = deque()
    queue.append((copy.deepcopy(board), is_bot_turn, None, 1))  # board_state, turn, first_move, depth
    best_score = float('-inf')
    best_move = None
    max_depth_reached = 1

    while queue:
        current_board, current_turn, first_move, depth = queue.popleft()
        max_depth_reached = max(max_depth_reached, depth)

        if game_over(current_board):
            score = evaluate(current_board, depth)
            if score > best_score or best_move is None:
                best_score = score
                best_move = first_move
            continue

        for i in range(3):
            for j in range(3):
                if current_board[i][j] == EMPTY:
                    new_board = copy.deepcopy(current_board)
                    new_board[i][j] = BOT if current_turn else HUMAN
                    move = (i, j)
                    next_first_move = first_move if first_move else move
                    queue.append((new_board, not current_turn, next_first_move, depth + 1))

    print(f"Max depth reached: {max_depth_reached}")
    return best_score, best_move


# Main game loop
def main():
    board = create_board()
    print("Welcome to X&O!")
    print_board(board)

    while not game_over(board):
        # Human move
        while True:
            try:
                move = input("Enter your move (row,col): ")
                x, y = map(int, move.strip().split(','))
                if board[x][y] == EMPTY:
                    board[x][y] = HUMAN
                    break
                else:
                    print("Cell is not empty!")
            except:
                print("Invalid input. Format should be: row,col (e.g., 1,2)")

        print_board(board)
        if game_over(board):
            break

        # Bot move
        print("Bot is thinking...")
        _, move = dfs(board, True)
        if move:
            board[move[0]][move[1]] = BOT
            print(f"Bot plays at: {move[0]},{move[1]}")
        print_board(board)

    # Game result
    if is_winner(board, HUMAN):
        print("You win! 🎉")
    elif is_winner(board, BOT):
        print("Bot wins! 🤖")
    else:
        print("It's a draw! 🤝")


if __name__ == "__main__":
    main()
