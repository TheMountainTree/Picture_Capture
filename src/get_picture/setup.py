from setuptools import find_packages, setup

package_name = 'get_picture'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='themountaintree',
    maintainer_email='wang2519480440@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'get_picture = get_picture.get_picture_node:main',
            'get_kinect2_node = get_picture.get_kinect2_node:main',
            'dual_cam_view = get_picture.dual_cam_view:main',
            'DualCameraSaveNode = get_picture.DualCameraSaveNode:main',
            'DualCameraKinectNode = get_picture.DualCameraKinectNode:main',
            'aruco_pose_node = get_picture.aruco_pose_node:main',
            'DualCameraOrbbecNode = get_picture.DualCameraOrbbecNode:main',
            'hand_eye_calib_node = get_picture.hand_eye_calib_node:main',
        ],
    },
)
