#!/bin/bash

IMAGE="registry.rcp.epfl.ch/nlp/ismayilz/cpo"
COMMAND=$1
CLUSTER=rcp-caas-prod
N_GPUS=1
N_CPUS=8
JOB_PREFIX="project-cpo"
JOB_SUFFIX="0"
MEMORY="64G"
GPU_MEMORY="32G"
VERBOSE=1
RUN_COMMAND=""
INTERACTIVE_RUN_COMMAND="bash startup.sh"
TRAIN_RUN_COMMAND="bash entrypoint.sh"
N_GPUS_SET=1
NODE_POOL="default"
DELETE_JOBS=0
ENTRYPOINT_ARGS=""
PREEMPTIBLE=0
PROJECT="nlp-ismayilz"
shift 1

while getopts m:l:c:g:ps:u:r:n:e:dq opt; do
	case ${opt} in
		m)
			MEMORY=${OPTARG}
			;;
		l)
			CLUSTER=${OPTARG}
			;;
		c)
			N_CPUS=${OPTARG}
			;;
		g)
			number_re='^[0-9]+$'
			if [[ ${OPTARG} =~ $number_re ]] ; then
				N_GPUS=${OPTARG}
			else
				GPU_MEMORY=${OPTARG}
				N_GPUS_SET=0
			fi
			;;
		p)
			PREEMPTIBLE=1
			;;
		s)
			JOB_SUFFIX=${OPTARG}
			;;
		u)
			GPU_MEMORY=${OPTARG}
			;;
		r)
			RUN_COMMAND=${OPTARG}
			;;
		n)
			NODE_POOL=${OPTARG}
			;;
		e)
			ENTRYPOINT_ARGS=${OPTARG}
			;;
		d)
			DELETE_JOBS=1
			;;
		q)
			VERBOSE=0
			;;
		?)
			echo unexpected option: ${opt}
			;;
	esac
done

JOB_NAME=${JOB_PREFIX}-${JOB_SUFFIX}

if [ "$VERBOSE" == 1 ]; then
	echo main command: ${COMMAND}
	echo image: ${IMAGE}
	echo cluster: ${CLUSTER}
	echo job name: ${JOB_NAME}
	echo "--------------------------------"
	
	echo cpus: ${N_CPUS}
	echo memory: ${MEMORY}

 	if [ "$N_GPUS_SET" == 1 ]; then
		echo gpus: ${N_GPUS}
	else
		echo gpu memory: ${GPU_MEMORY}
	fi

	if [ "$NODE_POOL" != "" ]; then
		echo node pool: ${NODE_POOL}
	fi

	if [ "$DELETE_JOBS" == 1 ]; then
		echo delete existing job: true
	fi

	if [ "$ENTRYPOINT_ARGS" != "" ]; then
		echo entrypoint args: ${ENTRYPOINT_ARGS}
	fi

	if [ "$PREEMPTIBLE" == 1 ]; then
		echo preemptible: true
	fi

	echo "--------------------------------"
fi

echo "using cluster: $CLUSTER"
runai config cluster $CLUSTER

echo "using project: $PROJECT"
runai config project $PROJECT

if [ "$DELETE_JOBS" == 1 ]; then
	echo "Deleting existing job $JOB_NAME"
	runai delete job $JOB_NAME
fi

echo "--------------------------------"

GPU_ARGS=""
if [ "$N_GPUS_SET" == 1 ]; then
	if [[ "$N_GPUS" > 0 ]]; then
		GPU_ARGS="--gpu $N_GPUS"
	fi
else
	GPU_ARGS="--gpu-memory $GPU_MEMORY"
fi

NODE_POOL_ARGS=""
if [ "$NODE_POOL" != "" ]; then
	NODE_POOL_ARGS="--node-pool $NODE_POOL"
fi

PREEMPTIBLE_ARGS=""
if [ "$PREEMPTIBLE" == 1 ]; then
	PREEMPTIBLE_ARGS="--preemptible"
fi

# Run this for train mode
if [ "$COMMAND" == "run" ]; then
	echo "Job [$JOB_NAME]"

	if [ "$RUN_COMMAND" == "" ]; then
		RUN_COMMAND="$TRAIN_RUN_COMMAND $ENTRYPOINT_ARGS"
	fi

	echo "Running command: $RUN_COMMAND"

	runai submit $JOB_NAME \
		-i $IMAGE \
		--cpu $N_CPUS \
		--memory $MEMORY \
		--large-shm \
		$GPU_ARGS \
		$NODE_POOL_ARGS \
		--pvc nlp-scratch:/mnt/scratch \
		--command -- $RUN_COMMAND
	exit 0
fi

# Run this for interactive mode
if [ "$COMMAND" == "run_bash" ]; then
	echo "Job [$JOB_NAME]"

	if [ "$RUN_COMMAND" == "" ]; then
		RUN_COMMAND="$INTERACTIVE_RUN_COMMAND"
	fi

	echo "Running command: $RUN_COMMAND"

	runai submit $JOB_NAME \
		-i $IMAGE \
		--cpu $N_CPUS \
		--memory $MEMORY \
		$GPU_ARGS \
		$NODE_POOL_ARGS \
		--pvc nlp-scratch:/mnt/scratch \
		--interactive \
		--attach \
		$PREEMPTIBLE_ARGS \
		--command -- $RUN_COMMAND
	exit 0
fi

if [ "$COMMAND" == "log" ]; then
	runai logs $JOB_NAME -f
	exit 0
fi

if [ "$COMMAND" == "stat" ]; then
	runai describe job $JOB_NAME 
	exit 0
fi

if [ "$COMMAND" == "del" ]; then
	runai delete job $JOB_NAME
	exit 0
fi

if [ $? -eq 0 ]; then
	runai list job
fi
