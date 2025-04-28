# Postgres

## Backup and Restore

We can use `pg_dumpall` to dump the instance:

```shell
sudo -u postgres pg_dumpall | gzip > all_databases.sql.gz
```

For single database, we can use `pg_dump`:

```shell
sudo -u postgres pg_dump -Fc mydatabase > mydatabase.dump
```

create a cron job

```shell
sudo visudo -f /etc/sudoers.d/anyinlover
# anyinlover ALL=(ALL) NOPASSWD: /usr/bin/pg_dumpall
cron -e
# 0 2 * * * sudo -u postgres pg_dumpall | gzip > /mnt/data/postgres_backup/$(date +\%Y\%m\%d\%H\%M\%S)_all_databases.sql.gz
# 0 3 * * * find /mnt/data/postgres_backup -type f -name "*.sql.gz" -mtime +7 -exec rm -f {} \;
```
