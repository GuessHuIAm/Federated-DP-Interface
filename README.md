# Federated Learning with Differential Privacy Simulator

## Quick Start

### Backend (FastAPI)

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Backend runs at `http://localhost:8000`
- API docs at `http://localhost:8000/docs`

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

- Frontend runs at `http://localhost:3000`
