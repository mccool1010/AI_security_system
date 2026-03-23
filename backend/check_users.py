from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['security_system']
users = list(db['users'].find({}))
print(f'Enrolled users: {len(users)}')
for u in users:
    print(f'  - {u.get("name")} (samples: {u.get("samples", "N/A")})')
