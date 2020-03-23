USER=$(shell whoami)
.PHONY: build
build: 
	docker build -t cvisionai/keras-retinanet:$(USER) -f docker/Dockerfile . || exit 255

ifeq ($(work_dir), )
extra_mounts=
else
extra_mounts=-v $(work_dir):/working --env deploy_dir=/working/deploy
endif
container_name=retinanet_$(USER)
ifeq ($(data_dir), )
else
extra_mounts+=-v $(data_dir):/data
container_name=retinanet_$(USER)_$(shell basename $(data_dir))
endif
ifndef retinanet_gpu
retinanet_gpu=0
endif
ifeq ($(shell uname -p), aarch64)
docker_cmd=nvidia-docker run
else
docker_cmd=docker run --gpus device=$(retinanet_gpu)
endif
ifndef retinanet_container_name
retinanet_container_name=$(container_name)_gpu_$(retinanet_gpu)
endif
.PHONY: dev_bash
dev_bash:
	$(docker_cmd) --name $(retinanet_container_name) --rm -ti --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v `pwd`/../deploy_python:/deploy_python $(extra_mounts) cvisionai/keras-retinanet:$(USER)
publish:
	docker tag cvisionai/keras-retinanet:$(USER) cvisionai/keras-retinanet:latest
	docker push cvisionai/keras-retinanet:latest
