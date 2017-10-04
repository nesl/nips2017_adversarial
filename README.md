## NIPS 2017 Adversarial Attacks and Defenses Competition Submission

This is the submissions of the **UCNesl** team during the [Competition on Adversarial Examples and Defenses](https://www.kaggle.com/nips-2017-adversarial-learning-competition) which is held as a part of NIPS'17 conference.


## Team members
* Moustafa Alzantot [(malzantot)](https://github.com/malzantot) - Leader for targeted attack.
* Yash Sharma [(ysharma1126)](https://github.com/ysharma1126) - Leader for non-targeted attack.
* Supriyo Charkaborty [(supriyogit)](https://github.com/supriyogit)
* Tianwei Xing [(TianweiXing)](https://github.com/TianweiXing)
* Sikai Yin [(Sikai Yin)](https://github.com/sikaiyin)
* [Prof. Mani Srivastava](http://nesl.ee.ucla.edu/people/1)

## Installation

### Prerequisites:

The following software is required to run this package:

* Python 2.7 with [Numpy](http://www.numpy.org/) and [Pillow](https://python-pillow.org/) Packages.
* [Docker] (https://www.docker.com/) or [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) for GPU support.

### Instructions:
* Download and extract the competition toolkit.
* Under the competition toolkit run the script `./download_data.sh` to download the development dataset.
* Copy the attacks from this repo (e.g. ucnesl_targeted, ucnesl_nontargeted to the appropriate folders under the competition toolkit directory (e.g. sample_targeted, and sample_attacks for targeted and non-targeted attacks respectively).
* Under each attack folder, run the `./download_checkpoints.sh` script to download the models checkpoints.
* Finally, you can run and evaluate the attacks by using `run_attacks_and_defenses.sh` script in the competition toolkit directory.

***Note*** For GPU-support, you should edit the the line #43 of the `run_attacks_and_defenses.sh` to add the following argument `--gpu` to the python command.

