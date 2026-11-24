\getenv monitoring_db POSTGRES_MONITORING_DB

SELECT format('CREATE DATABASE %I OWNER %I', :'monitoring_db', current_user)
WHERE NOT EXISTS (
    SELECT 1 FROM pg_database WHERE datname = :'monitoring_db'
)
\gexec
