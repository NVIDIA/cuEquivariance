import glob
import json
import shutil

import nox


@nox.session(python=["3.12"])
def third_party_attributions(session: nox.Session):
    """Generate third-party software license attributions."""
    session.install("pytest")
    session.install("torch")
    session.install("jax")
    session.install("../cuequivariance")
    session.install("../cuequivariance_jax")
    session.install("../cuequivariance_torch")
    # session.install("../cuequivariance_warp")
    session.install("pip-licenses")
    session.run(
        "pip-licenses",
        "--with-urls",
        "--ignore-packages",
        "cuequivariance",
        "cuequivariance_jax",
        "cuequivariance_torch",
        # "cuequivariance_warp",
        "pip-licenses",
        "prettytable",
        "wcwidth",
        "setuptools",
        "--format=json",
        "--with-license-file",
        "--with-system",
        "--output-file=Third_party_attr.json",
    )
    shutil.rmtree("../cuequivariance/build", ignore_errors=True)
    shutil.rmtree("../cuequivariance_jax/build", ignore_errors=True)
    shutil.rmtree("../cuequivariance_torch/build", ignore_errors=True)

    with open("Third_party_attr.json") as file:
        data = json.load(file)

    with open("Third_party_attr.txt", "w") as output_file:
        print("# Third Party Software Attributions\n\n")
        for item in data:
            if "nvidia" in item["Name"].lower():
                continue

            if item["License"] == "UNKNOWN":
                # Try to infer license from license text
                license_text = item.get("LicenseText", "")
                if "Apache License" in license_text:
                    item["License"] = "Apache License 2.0"
                elif (
                    "HISTORY OF THE SOFTWARE" in license_text
                    and "Python Software Foundation" in license_text
                ):
                    item["License"] = "Python Software Foundation License"
                else:
                    raise ValueError(f"License not found for {item['Name']}")

            output_file.write(
                (
                    f"Name: {item['Name']}\n "
                    f"Version: {item['Version']}\n "
                    f"License: {item['License']} \n "
                    f"URL: {item['URL']}\n "
                    f"License Text:\n"
                    f"{item['LicenseText']}\n"
                    "*******************\n\n"
                )
            )
            print(item["Name"], "--", item["License"], "--", item["URL"])

        # append files like .code/LICENSE.e3nn-jax
        code_licenses = glob.glob(".code/LICENSE.*")

        for code_license in code_licenses:
            with open(code_license, "r") as file:
                output_file.write(f"License file: {code_license}\n")
                output_file.write("License Text:\n")
                output_file.write(file.read())
                output_file.write("\n*******************\n\n")
