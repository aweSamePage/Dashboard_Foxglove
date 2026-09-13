from setuptools import find_packages, setup


package_name = "roboracer_dashboard_foxglove"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/foxglove", ["foxglove/layout_ftg_debug.json"]),
        (f"share/{package_name}", ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="F1TENTH Team",
    maintainer_email="student@example.com",
    description="Foxglove Follow The Gap debug dashboard for F1TENTH Gym.",
    license="MIT",
    tests_require=["pytest"],
)
