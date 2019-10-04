"""

@author: j.v.griethuysen@nki.nl
LICENSE: BSD 3 Clause 2018
"""

import os
import json
import logging

from six.moves.urllib.request import Request, urlopen
from six.moves.urllib.error import URLError, HTTPError


class SlackHandler(logging.Handler):

  def __init__(self,
               channel=None,
               url='https://slack.com/api/chat.postMessage',
               api_key=None,
               bot_name=None,
               level=logging.NOTSET):
    super(SlackHandler, self).__init__(level)
    if channel is None:
      channel = os.environ.get('SLACK_USER_ID', '#general')
    self.channel = channel
    self.url = url
    if api_key is None:
      api_key = os.environ.get('SLACK_API_TOKEN')
    self.api_key = api_key
    self.bot_name = bot_name

    self.level_emojis = {
      'WARNING': ':anguished:',
      'ERROR': ':face_with_symbols_on_mouth:',
      'CRITICAL': ':skull_and_bones:'
    }

    assert self.channel is not None, 'A valid slack channel to post log records to is required.'
    assert self.api_key is not None, 'Need an API Key to be able to utilize Slack API'

  def emit(self, record):

    emoji = self.level_emojis.get(record.levelname, ':innocent:')

    data = {
      'text': self.format(record),
      'channel': self.channel,
      'as_user': self.bot_name is None,
      'icon_emoji': emoji
    }

    if self.bot_name is not None:
      data['username'] = self.bot_name

    req = Request(self.url, json.dumps(data).encode())
    req.add_header('content-type', 'application/json')
    req.add_header('Authorization', 'Bearer %s' % self.api_key)

    try:
      urlopen(req)
    except (URLError, HTTPError) as e:
      print(e.message)
