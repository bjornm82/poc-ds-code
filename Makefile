
build:
	docker build -t bjornmooijekind/test-model:latest .

push:
	docker push bjornmooijekind/test-model:latest

getdata:
	docker run -it bjornmooijekind/test-model:latest python3 getdata.py

train:
	docker run -it bjornmooijekind/test-model:latest python3 train.py
