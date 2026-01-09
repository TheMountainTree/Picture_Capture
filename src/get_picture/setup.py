from setuptools import setup

package_name = 'get_picture'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools', 'rclpy'],
    entry_points={
        'console_scripts': [
            'get_picture_node = get_picture.get_picture_node.get_picture_node:main',  # 指向你的 Python 文件
            'get_kinect2_node = get_picture.get_picture_node.get_kinect2_node:main',
            'dual_cam_view = get_picture.get_picture_node.dual_cam_view:main',
            'DualCameraSaveNode = get_picture.get_picture_node.DualCameraSaveNode:main',
        ],
    },
)
