
build:
	docker build -t model/base:latest .

push:
	docker push model/base:latest

getdata:
	docker run -it model/base:latest python3 getdata.py

train:
	docker run -it model/base:latest python3 train.py
