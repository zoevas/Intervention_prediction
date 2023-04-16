printf "Start Nightingale Patient predictive assessement deployment..\n"

IMAGE_NAME=predassessment
CONTAINER_NAME=predassessmentcontainer

HOST_PORT=8080
APP_PORT=8080

echo Building Docker image. TAG = "$IMAGE_NAME"
sudo docker build -t $IMAGE_NAME -f Dockerfile .

sudo docker run -p  $HOST_PORT:$APP_PORT  --name $CONTAINER_NAME $IMAGE_NAME

echo App successfully deployed
