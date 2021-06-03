# Ansible
# Author Jose Smith
# Start Date: 20210521
# End Date: 

## Notable Links
https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html \
https://galaxy.ansible.com/

## (1) Installing Ansible
### Get the proper PIP: 
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py` \
`export PATH="$HOME/.local/bin:$PATH"` \
`python3 get-pip.py --user`

### Install Ansible with pip: 
`python -m pip install --user ansible` \
## or install with: 
`sudo apt install ansible`

### Install Autocomplete: 
`sudo apt install python3-argcomplete` \
`sudo activate-global-python-argcomplete` \

## (2) Getting Started with Ansible
### The inventory file, roles and configuration 
The inventory file is by default the /etc/ansible/hosts file and by default you will have to have elevated access to change this file. \
The roles directory should be in the /etc/ansible folder. \
The ansible.cfg is available in the /etc/ansible folder for customization.\
### You can verify that the hosts were added correctly with the following command:
`ansible all -m ping -vvv`

### Playbooks
Playbooks can be stored anywhere and for this lab I will be installing them in my ~/Documents/ansible folder in the *.yaml format\ 
Once you have a playbook you can run the following command to connect to the machines:\
`ansible-playbook sampleplaybook.yaml` \
or\
`ansible-playbook --private-key=/home/<user_id>/.ssh/id_rsa sampleplaybook.yaml -u <user_id> --ask-pass`

## (3) Working with Ansible
### hosts and variables in ansible
You can target hosts in the host file by putting groups in [groups] or even target groups using a : to show that it is a collection of groups [maingroup:subgroup] or even throw in localhost as a host in there. This allows for targetting variables to be placed against these hosts. \
Here is a list of the default modules in ansible:\
https://docs.ansible.com/ansible/2.9/modules/list_of_all_modules.html\
Here is a sample command to run your first module:
`ansible localhost -m find -a "paths=~/Downloads file_type=file"`\
The sample command should list all the files in the home Downloads directory and all the permissions for it. Be warned that it is a lot of information and if you have more than a few files it will fill your screen. For my test I ran `touch Success.txt` and `touch Failure.txt` to populate that directory and made sure nothing else was in it.\

