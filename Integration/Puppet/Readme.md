Puppet
Author Jose Smith
Start Date: 20210608
End Date: 

# Notable Information
## Links to resources
https://puppet.com/docs/
https://www.godaddy.com/garage/install-puppet-centos7/
## Troubleshooting:
 

# (1) Setting Up a Dev Environment
## Create a sandbox
### Install Virtual Box
https://www.virtualbox.org/wiki/Downloads \
https://itsfoss.com/install-virtualbox-ubuntu/ \
`sudo apt install virtualbox`
### Install Vagrant:
https://www.vagrantup.com/
`curl -O https://releases.hashicorp.com/vagrant/2.2.9/vagrant_2.2.9_x86_64.deb`\
`sudo apt install ./vagrant_2.2.9_x86_64.deb`\
`vagrant --version`
## Install your Puppet master
### Create the VM
Review your vagrantfile in exercise01 and from that directory do a`vagrant up`\
Then run `vagrant ssh ` from the same folder.\
### Install the puppet server
`sudo su -`\
`rpm -ivh https://yum.puppetlabs.com/el/7/products/x86_64/puppetlabs-release-7-11.noarch.rpm`\
`yum install -y puppetserver vim git`\
`vim /etc/sysconfig/puppet` to change the `JAVA_ARGS` to `512m` as we don't need a large server\
`systemctl start  puppetserver`\
`systemctl enable  puppetserver`\
`vim /etc/puppet/puppet.conf` and add `server = master.puppet.vm` to the `[agent]` section.
### Update ruby and install ruby gems
`yum -y install centos-release-scl-rh centos-release-scl`\
`sed -i -e "s/\]$/\]\npriority=10/g" /etc/yum.repos.d/CentOS-SCLo-scl.repo`\
`sed -i -e "s/\]$/\]\npriority=10/g" /etc/yum.repos.d/CentOS-SCLo-scl-rh.repo`\
`yum --enablerepo=centos-sclo-rh -y install rh-ruby27`\
Add `rh-ruby27.sh` to `/etc/profile.d/rh-ruby27.sh`\
`scl enable rh-ruby27 bash`\
`gem install r10k`
### Verify install
`puppet agent -t`, this should take about a second and do Info messages followed by a notice for how long it took to install. Look for `master.puppet.vm` to be listed in the catalog entry.
## Version Control
## Set up a control repo

# (2) First Steps with Puppet
## Built-in resource types
## Manage a file in site.pp
## Classes
## Introduction to the Forge
## The NGINX module
## Editing the Puppetfile
## Roles and profiles

# (3) Managing More Roles
## Manage more nodes
## Expand site.pp
## Connect agent nodes to the master
## Orchestration in Puppet
## Understand the Puppet run
## Facter
## Installing SSH and Adding Hosts

# (4) Modules
## Write Manually
## Write the Code
## Test your module
## Get the order right
## Use parameters
## Templates

# (5) Next Steps