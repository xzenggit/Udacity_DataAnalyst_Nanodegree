# 5) Using pymongo to connect MongoDB
# Before this step, we need to
# Start a running MongoD instance at shell: `mongod`
# Import the JSON file into MongoDB at shell:
# `mongoimport --db OpenStreetMap --collection Raleigh_NC --drop --file raleigh_north-carolina.osm.json `

from pymongo import MongoClient
import pprint

client = MongoClient('localhost', 27017)
db = client['OpenStreetMap']
collection = db['Raleigh_NC']


# 6) Count different categories
print 'Total number of documents: ', collection.find().count()
print 'Number of node:', collection.find({'type': 'node'}).count()
print 'Number of way:', collection.find({'type': 'way'}).count()


# 7) Top 10 contributors
a = collection.aggregate([{"$group": {"_id": "$created.user", "count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])
print 'Top 10 contributors:'
pprint.pprint(list(a))

# 8) Top 10 contributors with UID (check consistentency)
a = collection.aggregate([{"$group": {"_id": "$created.uid", "count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])
print 'Top 10 IDs:'
pprint.pprint(list(a))


# 9) Number of users appearing only once (having 1 post)
b = collection.aggregate([{"$group":{"_id":"$created.user", "count":{"$sum":1}}}, {"$group":{"_id":"$count", "num_users":{"$sum":1}}}, {"$sort":{"_id":1}}, {"$limit":1}])
print 'Number of users appearing only once (having 1 post):'
print list(b)


# 10) No.1 appearing place names
a = collection.aggregate([{"$match":{"name":{"$exists":1}}}, {
            "$group":{"_id":"$name","count":{"$sum":1}}
        }, {"$sort":{"count":-1}}, {"$limit":1}])
print 'No.1 appearing place name:'
pprint.pprint(list(a))