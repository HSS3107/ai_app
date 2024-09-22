#!/bin/bash
service ssh start

python /code/manage.py migrate
echo "Migrate completed"
python /code/manage.py collectstatic --noinput
echo "Collect static completed"

# Ensure the script exits if any command fails
set -e

# Execute the command passed as argument (in your case, uWSGI)
exec "$@"
