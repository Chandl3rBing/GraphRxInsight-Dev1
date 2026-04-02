# GraphRxInsight

Drug-drug interaction prediction app with side-effect context and GNN/GAT feature visualization.

## Instant Public Link (Temporary)

You can expose the local running app immediately:

```bash
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -R 80:localhost:5001 nokey@localhost.run
```

The command prints an `https://...lhr.life` URL that users can open.

## Deploy (Docker)

From the project root:

```bash
docker compose up -d --build
```

Open:

```text
http://localhost:5001
```

## Stop

```bash
docker compose down
```

## Notes

- Backend runs with `gunicorn` on port `5001`.
- The backend serves the built React frontend from `frontend/build`.
- `models/` and `DATASETS/` are mounted so feedback samples and dynamic model updates persist.
- `backend/app.py` can load features from either:
  - `DATASETS/processed/unified_drug_features.csv`
  - `DATASETS/processed/unified_drug_features.csv.gz` (recommended for Git hosting limits)

## Permanent Cloud Deploy (Render)

This repo includes `render.yaml` and `Dockerfile` for one-click Render deployment.

1. Push this repo to GitHub (with `render.yaml`, Dockerfile, backend/frontend changes, and `unified_drug_features.csv.gz`).
2. In Render Dashboard, choose **New +** -> **Blueprint**.
3. Select this GitHub repository and deploy.
4. After first deploy, share the generated `https://<service>.onrender.com` URL with users.
