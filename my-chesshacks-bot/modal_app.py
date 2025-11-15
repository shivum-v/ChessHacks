import modal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import chess
import time

# Create a Modal app
app = modal.App("chesshacks-bot")

# Define the image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "torchaudio")
    .add_local_file("requirements.txt", "/requirements.txt", copy=True)
    .run_commands("pip install -r /requirements.txt")
    .add_local_dir("src", "/src")
    .add_local_file("trained_model.pt", "/src/trained_model.pt")  # Include trained model
)

@app.function(
    image=image,
    allow_concurrent_inputs=True,
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Create and return the FastAPI app for Modal to serve."""
    import sys
    sys.path.insert(0, "/")
    
    from src.utils import chess_manager
    from src import main
    
    web_app = FastAPI()

    @web_app.post("/")
    async def root():
        return JSONResponse(content={"running": True})

    @web_app.post("/move")
    async def get_move(request: Request):
        try:
            data = await request.json()
        except Exception as e:
            return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

        if ("pgn" not in data or "timeleft" not in data):
            return JSONResponse(content={"error": "Missing pgn or timeleft"}, status_code=400)

        pgn = data["pgn"]
        timeleft = data["timeleft"]  # in milliseconds

        chess_manager.set_context(pgn, timeleft)
        print("pgn", pgn)

        try:
            start_time = time.perf_counter()
            move, move_probs, logs = chess_manager.get_model_move()
            end_time = time.perf_counter()
            time_taken = (end_time - start_time) * 1000
        except Exception as e:
            time_taken = (time.perf_counter() - start_time) * 1000
            return JSONResponse(
                content={
                    "move": None,
                    "move_probs": None,
                    "time_taken": time_taken,
                    "error": "Bot raised an exception",
                    "logs": None,
                    "exception": str(e),
                },
                status_code=500,
            )

        # Confirm type of move_probs
        if not isinstance(move_probs, dict):
            return JSONResponse(content={"move": None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

        for m, prob in move_probs.items():
            if not isinstance(m, chess.Move) or not isinstance(prob, float):
                return JSONResponse(content={m: None, "move_probs": None, "error": "Failed to get move", "message": "Move probabilities is not a dictionary"}, status_code=500)

        # Translate move_probs to Dict[str, float]
        move_probs_dict = {move.uci(): prob for move, prob in move_probs.items()}

        return JSONResponse(content={"move": move.uci(), "error": None, "time_taken": time_taken, "move_probs": move_probs_dict, "logs": logs})

    return web_app

