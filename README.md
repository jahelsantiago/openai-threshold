This project is a POC for finding the ideal threshold in a semantic search.

To run it, please add a .env at the root of the project with the following values:

```shell
# Once you add your API key below, make sure to not share it with anyone! The API key should remain private.
OPENAI_API_KEY=<your-api-key-here>

# Postgres 
DATABASE_HOST=localhost
DATABASE_USERNAME=<your-db-user-here>
DATABASE_PASSWORD=<your-db-password-here>
DATABASE_PORT=5432
DATABASE_NAME=<your-db-name-here>
```

You can run this on your local DB or using an ssh connection to your staging db:
```shell
ssh -i ./your-key-file.pem ubuntu@ec2-x-xxx-xx-xx.compute-1.amazonaws.com -L 5432:<db-url>:5432 -N
```
