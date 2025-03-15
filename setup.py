from setuptools import find_packages, setup

package_name = 'cdrr_adaptive_ros2'

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
    maintainer='awear',
    maintainer_email='realfcn@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'adaptive_learning = cdrr_adaptive_ros2.adaptive_learning:main',
            'simulation_env = cdrr_adaptive_ros2.simulation_env:main',
            'visualization = cdrr_adaptive_ros2.visualization:main',
        ],
    },
)
