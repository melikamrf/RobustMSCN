#!/bin/sh

check_pg_ready() {
  pg_isready -U $POSTGRES_USER > /dev/null 2>&1
}

echo "Waiting for PostgreSQL to be ready..."

# Loop until PostgreSQL is ready
while ! check_pg_ready; do
  echo "PostgreSQL is not ready yet. Retrying in 1 second..."
  sleep 1
done

echo "PostgreSQL is ready!"

createdb -U $POSTGRES_USER imdb

wget -O /var/lib/postgresql/pg_imdb.tar https://www.dropbox.com/s/vq1owleo9nuyxdf/pg_imdb.tar.gz?dl=1
tar xf /var/lib/postgresql/pg_imdb.tar -C /var/lib/postgresql/

psql -U $POSTGRES_USER -c "CREATE ROLE imdb WITH SUPERUSER LOGIN;"
psql -U $POSTGRES_USER -c "ALTER USER imdb PASSWORD 'password'"

psql -U $POSTGRES_USER -c "CREATE ROLE postgres WITH SUPERUSER LOGIN;"
psql -U $POSTGRES_USER -c "ALTER USER postgres PASSWORD 'password'"

#psql -d imdb -U $POSTGRES_USER -c "SHOW max_wal_size";
pg_restore -v -d imdb -U $POSTGRES_USER /var/lib/postgresql/pg_imdb
