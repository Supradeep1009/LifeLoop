"""
LifeLoop Server Entry Point
Starts the FastAPI/Uvicorn server for the LifeLoop OpenEnv environment.
"""

import uvicorn


def main():
    """Run the LifeLoop server."""
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
