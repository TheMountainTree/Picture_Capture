016244444847 # 光的

006799463647 # 带支架的

ros2 run kinect2\_bridge kinect2\_bridge\_node --ros-args -p sensor:="'006799463647'" -p base\_name:=kinect2\_main

ros2 run kinect2\_bridge kinect2\_bridge\_node --ros-args -p sensor:="'016244444847'" -p base\_name:=kinect2\_assi

ros2 launch kinect2\_bridge kinect2\_bridge\_launch.yaml sensor:="'006799463647'" base\_name:=kinect2\_main #带有点云

ros2 launch kinect2\_bridge kinect2\_bridge\_launch.yaml sensor:="'016244444847'" base\_name:=kinect2\_assi #带有点云
