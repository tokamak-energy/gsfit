import pprint
from importlib import metadata
from importlib.metadata import version

from .. import version_storage


def get_python_library() -> str:
    library_list = []

    # Retrieve all of the "distributions" and store as list
    distribution_list = [distribution.name for distribution in metadata.distributions()]
    # Remove any "None"
    distributions_sorted = sorted(list(set([x for x in distribution_list if x is not None])))
    # The "set" function will reurn unique values.
    # distributions_sorted = distribution_list.sort()

    for distribution in distributions_sorted:
        this_entry = {
            "package": distribution,
            "version": version(distribution),
        }
        library_list.append(this_entry)
    library_str = pprint.pformat(library_list)

    version_storage.__python_library__ = library_str

    return library_str


# import pkg_resources # BUXTON: this is deprecated!

# def get_python_library():
#     library_list = []
#     for package in pkg_resources.working_set:
#         package_name = package.project_name
#         this_entry = {
#             'package': package_name,
#             'version': version(package_name)
#         }
#         library_list.append(this_entry)
#     library_str = pprint.pformat(library_list)

#     version_storage.__python_library__ = library_str

#     return library_str
