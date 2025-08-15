#测试向量数据库是否连接成功
from pymilvus import connections

connections.connect(
    host="localhost",
    port="19530",
    user="",
    password=""
)
print(connections.list_connections())  # 输出连接信息
