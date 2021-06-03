Ansible
Author Jose Smith
Start Date: 20210602
End Date: 

# Notable Information
## Links to resources
https://docs.ansible.com/ansible/latest/installation_guide/intro_installation.html \
https://galaxy.ansible.com/\
https://github.com/web-id-fr/mario
## Connection Issues:
In ansible There is no option to store passphrase-protected private key\
For that we need to add the passphrase-protected private key in the ssh-agent\
Start the ssh-agent in the background.\
`eval "$(ssh-agent -s)"`\
Add SSH private key to the ssh-agent\
`ssh-add ~/.ssh/id_rsa`\
Now try running ansible-playbook and ssh to the hosts.

# (1) Installing Ansible
## Get the proper PIP: 
`curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py` \
`export PATH="$HOME/.local/bin:$PATH"` \
`python3 get-pip.py --user`

## Install Ansible with pip: 
`python -m pip install --user ansible` \
## or install with: 
`sudo apt install ansible`

## Install Autocomplete: 
`sudo apt install python3-argcomplete` \
`sudo activate-global-python-argcomplete` \

# (2) Getting Started with Ansible
## The inventory file, roles and configuration 
The inventory file is by default the `/etc/ansible/hosts` file and by default you will have to have elevated access to change this file. \
The roles directory should be in the `/etc/ansible` folder. \
The `ansible.cfg` is available in the `/etc/ansible` folder for customization.\
## You can verify that the hosts were added correctly with the following command:
`ansible all -m ping -vvv`

## Playbooks
Playbooks can be stored anywhere and for this lab I will be installing them in my` ~/Documents/ansible folder` in the `*.yaml` format\ 
Once you have a playbook you can run the following command to connect to the machines:\
`ansible-playbook sampleplaybook.yaml` \
or\
`ansible-playbook --private-key=/home/<user_id>/.ssh/id_rsa sampleplaybook.yaml -u <user_id> --ask-pass`

# (3) Working with Ansible
## hosts and variables in ansible
You can target hosts in the host file by putting groups in `[groups]` or even target groups using a `:` to show that it is a collection of groups `[subgroup:children]` where `subgroup` is the new variable or even throw in localhost as a host in there. This allows for targetting variables to be placed against these hosts. \
Here is a list of the default modules in ansible:\
https://docs.ansible.com/ansible/2.9/modules/list_of_all_modules.html\
Here is a sample command to run your first module:
`ansible localhost -m find -a "paths=~/Downloads file_type=file"`\
The sample command should list all the files in the home Downloads directory and all the permissions for it. Be warned that it is a lot of information and if you have more than a few files it will fill your screen. For my test I ran `touch Success.txt Failure.txt` to populate that directory and made sure nothing else was in it.
## Working with Playbooks
We create the `first.yml` file with a ping test and install the stress application. Then run that playbook with `ansible-playbook first.yml'.\
For the second test we are creating `second.yml` to combine both the playbook that we had created earlier and that command line that we used earlier to do a ping test and then show the files.\ 
Run the playbook normal and with `-v` and note the output difference:\
`ansible-playbook second.yml`\
`ansible-playbook second.yml -v`\
As you can see the output was a bit of a mess so we are going to clean up the ansible.cfg to use the YAML callback plugin and use stdout_callback when running ad-hoc commands.\
Add the following to `sudo vim /etc/ansible/ansible.cfg`, for me it was around line 74:\
`# Use the YAML callback plugin.`\
`stout_callback = yaml`\
`# Use the stoud_callback when running ad-hoc commands.`\
`bin_ansible_callbacks = True`\
`interpreter_python = auto_silent`

# (4)) What Can Ansible Do for You?
## Ansible for system configuration management
In this section we are going to create an NTP state and ensure that all the servers are able to be kept accurate:\
`sudo ansible-playbook ntp.yml`\
## Reacting to Change with Ansible
We can update the `sampleplaybook.yml` to add `changed_when` and start exploring `handlers` that will automate tasks.