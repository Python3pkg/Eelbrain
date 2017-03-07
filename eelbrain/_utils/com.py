# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Internet communication utilities"
from __future__ import print_function
from distutils.version import LooseVersion
from email.mime.text import MIMEText
import keyring
import smtplib
import socket
import sys
import traceback
import xmlrpclib

from .system import caffeine
from . import ui


NOOB_DOMAIN = "Eelbrain"
NOOB_ADDRESS = 'n00b.eelbrain@gmail.com'


def check_for_update():
    """Check whether a new version of Eelbrain is available

    Prints a message if an update is available on the Python package index, does
    nothing otherwise.
    """
    current = sys.modules['eelbrain'].__version__
    if current == 'dev':
        return print("Using Eelbrain development version")
    pypi = xmlrpclib.ServerProxy('https://pypi.python.org/pypi')
    versions = pypi.package_releases('eelbrain')
    newest = versions[-1]
    if LooseVersion(newest) > LooseVersion(current):
        print("New Eelbrain version available: %s (currently installed is %s)" %
              (newest, current))


def get_smtpserver(password, new_password=False):
    smtpserver = smtplib.SMTP('smtp.gmail.com', 587)
    smtpserver.ehlo()
    smtpserver.starttls()
    while True:
        try:
            smtpserver.login(NOOB_ADDRESS, password)
            if new_password:
                keyring.set_password(NOOB_DOMAIN, NOOB_ADDRESS, password)
            return smtpserver
        except smtplib.SMTPAuthenticationError:
            password = ui.ask_str("Eelbrain notifier password invalid. Please "
                                  "enter valid password.", "Notifier Password")
            if password:
                new_password = True
            else:
                raise


def send_email(to, subject, body, password):
    """Send an email notification"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    host = socket.gethostname()
    msg['From'] = 'Eelbrain on %s <%s>' % (host, NOOB_ADDRESS)
    msg['To'] = to

    smtpserver = get_smtpserver(password)
    smtpserver.sendmail(NOOB_ADDRESS, to, msg.as_string())
    smtpserver.close()


class Notifier(object):
    """
    A notification email sender supporting ``with`` statements

    Parameters
    ----------
    to : str
        Email address of the recipient.
    name : str
        Name of the job (will be included in subject line).
    crash_info_func : None | callable
        Will be called upon crash to produce a string that will be included
        in the crash report.
    debug : bool
        If the task crashes, start pdb instead of exiting.

    Examples
    --------
    To receive a message after a task has been executed:

    >>> notifier = Notifier('me@somewhere.com')
    >>> with notifier:
    ...     do_task()
    ...

    """
    def __init__(self, to, name='job', crash_info_func=None, debug=True):
        # get the password
        password = keyring.get_password(NOOB_DOMAIN, NOOB_ADDRESS)
        if password is None:
            password = ui.ask_str("Please enter the Eelbrain notifier "
                                  "password.", "Notifier Password")
            # test it
            print("Validating password...")
            smtpserver = get_smtpserver(password, True)
            smtpserver.close()

        self.to = to
        self.name = name
        self.crash_info_func = crash_info_func
        self.debug = debug
        self._password = password

    def __enter__(self):
        print("Notification enabled for %s..." % self.name)
        caffeine.__enter__()
        return self

    def __exit__(self, type_, value, traceback_):
        host = socket.gethostname()
        caffeine.__exit__(type_, value, traceback_)
        if isinstance(value, Exception):
            error = type_.__name__
            temp = '{host} encountered {error}: {value} in {task}'
            event = temp.format(host=host, error=error, value=value,
                                task=self.name)
            info = []

            # traceback
            tb_items = traceback.format_tb(traceback_)
            error_message = "%s: %s\n" % (error, value)
            tb_items.append(error_message)
            tb_str = '\n'.join(tb_items)
            info.append(tb_str)

            # object info
            if self.crash_info_func:
                info.extend(self.crash_info_func())

            self.send(event, info)

            # drop into pdb
            if self.debug:
                traceback.print_exc()
                print('')
                try:
                    import ipdb as pdb
                except ImportError:
                    import pdb
                pdb.post_mortem(traceback_)
        else:
            event = '{host} finished {task}'.format(host=host, task=self.name)
            self.send(event)

    def send(self, subject, info=()):
        """Send an email message

        Parameters
        ----------
        subject : str
            Email subject line.
        info : str | list of str
            Email body; if a list, successive entries are joined with three
            line breaks.
        """
        if isinstance(info, basestring):
            body = info
        else:
            body = '\n\n\n'.join(map(unicode, info))

        try:
            send_email(self.to, subject, body, self._password)
        except Exception as error:
            print ("Could not send email because an error occurred, skipping "
                   "notification. Check your internet connection.\n%s"
                   % str(error))


class NotNotifier(object):
    # Helper to raise proper error message when user has not set owner attribute
    def __enter__(self):
        raise AttributeError("The notifier is disabled because the .owner "
                             "attribute was not set")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
