# Frontend Dashboard

## Setup

```bash
cp .env.example .env
npm install
npm run dev
```

App runs on `http://localhost:3000`.

Configure API base URL in `.env`:

```bash
VITE_API_BASE_URL=http://YOUR_SERVER:8000
```

Login notes:
- The frontend no longer uses hardcoded demo users.
- Credentials are validated by backend endpoint `/api/v1/auth/login`.
- Users come from `app_users` table in central database.
