from py4j.java_gateway import JavaGateway

# Connect to the Java Gateway
gateway = JavaGateway()
ir_system = gateway.entry_point

#### CREATE INDEX #####
# folderPath = "src/main/resources"
# ir_system.index(folderPath)
#### CREATE INDEX #####


#### SEARCH QUERY #####
query = "peisaje spectaculoase"
results = ir_system.search(query)

print(results)
#### SEARCH QUERY #####
