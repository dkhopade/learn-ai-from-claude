## Running locally
```bash
source venv/bin/activate
uvicorn main:app --reload
```

## Note on containers
Docker is blocked by corporate policy on this machine.
Will revisit containerization in Week 5 when deploying to OCI/OKE,
where containers can be built in CI/CD pipeline instead.
