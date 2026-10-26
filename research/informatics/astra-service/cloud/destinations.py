"""Fixed broker destinations. Never accept a caller-provided URL."""
DESTINATIONS={
 'production':('https://informatics-universal-one.universalmind.workers.dev','https://universalmind.tv'),
 'staging':('https://informatics-universal-one-staging.universalmind.workers.dev','https://staging.universalmind.tv'),
}

def destination(environment):
 if environment not in DESTINATIONS:raise ValueError('Unknown research environment')
 return DESTINATIONS[environment]
