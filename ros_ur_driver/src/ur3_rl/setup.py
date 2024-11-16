from setuptools import find_packages, setup

package_name = 'ur3_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'geometry_msgs',
        'std_srvs',
        'ur_msgs',
        'gymnasium',
        'torch',
        'skrl',
        'numpy',
    ],
    zip_safe=True,
    maintainer='rl-public',
    maintainer_email='sp.medina@javeriana.edu.co',
    description='RL environment and evaluation for UR3 robot',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'simple_controller = ur3_rl.simple_controller:main',        
            'evaluate_ur3 = ur3_rl.evaluate_ur3:main',
            'move_ur3_to_target = ur3_rl.move_ur3_to_target:main'
        ],
    },
)
