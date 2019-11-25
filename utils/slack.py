"""

@author: j.v.griethuysen@nki.nl
LICENSE: BSD 3 Clause 2018
"""

import datetime
import json
import logging
import os
import re
import time

from six.moves.urllib.request import Request, urlopen
from six.moves.urllib.error import URLError, HTTPError


class SlackHandler(logging.Handler):
  """
  This class can be used as a handler to a logger, enabling logging from a script to slack

  To work, a token is required. This can be generated
  `<here> https://api.slack.com/custom-integrations/legacy-tokens`_.
  Next, some variables need to be determined, for which this class provides some helper methods:


  """
  url = 'https://slack.com/api'
  _requests = 0
  _start = None

  def __init__(self,
               channel=None,
               api_key=None,
               bot_name=None,
               level=logging.NOTSET):
    super(SlackHandler, self).__init__(level)
    if channel is None:
      assert 'SLACK_USER_ID' in os.environ, 'Need either a channel, or Environment variable SLACK_USER_ID to work!'
      channel = os.environ['SLACK_USER_ID']
    self.channel = channel
    if api_key is None:
      api_key = os.environ.get('SLACK_API_TOKEN')
    self.api_key = api_key
    self.bot_name = bot_name

    self.level_emojis = {
      'WARNING': ':anguished:',
      'ERROR': ':face_with_symbols_on_mouth:',
      'CRITICAL': ':skull_and_crossbones:'
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
      if '%(levelname)' in self.bot_name:
        data['username'] = self.bot_name % {'levelname': record.levelname}
      else:
        data['username'] = self.bot_name

    try:
      self.do_request('chat.postMessage', self.api_key, data)
    except (URLError, HTTPError) as e:
      print('HTTP error! ' + e.read().decode(
        'utf-8'))  # Do not include a traceback, that's not interesting here (it's not a bug, but server complaining...)

  @classmethod
  def do_request(cls, cmd, token=None, data=None):
    now = datetime.datetime.now()
    if cls._start is None or cls._start < (now - datetime.timedelta(seconds=20)):
      cls._start = now
      cls._requests = 0
    elif cls._requests >= 20:
      print('Slack handler sleeping until 25s to prevent "Too-Many-Requests" HTTP exception')
      time.sleep(25 - (now - cls._start).seconds)
      cls._requests = 0

    cls._requests += 1

    if data is None:
      req = Request('%s/%s' % (cls.url, cmd))
    else:
      req = Request('%s/%s' % (cls.url, cmd), json.dumps(data).encode())

    req.add_header('content-type', 'application/json; charset=utf-8')
    if token is not None:
      req.add_header('Authorization', 'Bearer %s' % token)

    resp = urlopen(req)
    response_json = json.loads(resp.read().decode("utf-8"))
    resp.close()
    return response_json

  @classmethod
  def get_user(cls, api_token, email):
    try:
      users = cls.do_request('users.list', api_token)
    except (URLError, HTTPError) as e:
      print('HTTP error! ' + e.read().decode(
        'utf-8'))  # Do not include a traceback, that's not interesting here (it's not a bug, but server complaining...)
      return

    assert users['ok'], 'Non-OK response received: %s' % users
    for u in users['members']:
      if u['profile']['email'] == email:
        print('Found match for user %s (%s)' % (u['name'], u['real_name']))
        return u['id']
    print('No match made for email %s in %i users!' % (email, len(users['members'])))

  @classmethod
  def get_channel(cls, api_token, user='USLACKBOT'):
    try:
      im_conversations = cls.do_request('users.conversations?types=im', api_token)
    except (URLError, HTTPError) as e:
      print('HTTP error! ' + e.read().decode(
        'utf-8'))  # Do not include a traceback, that's not interesting here (it's not a bug, but server complaining...)
      return

    assert im_conversations['ok'], 'Non-OK response received: %s' % im_conversations
    for im in im_conversations['channels']:
      if im['user'] == user:
        return im['id']

    print('Slackbot channel not found! Checked %i im channels' % len(im_conversations['channel']))

  @classmethod
  def cleanup_channel(cls, api_token, channel_id, sender, date_threshold=None, text_regex=None, max_deletions=None):
    sender_pattern = None
    if sender is not None:
      sender_pattern = re.compile(sender)

    pattern = None
    if text_regex is not None:
      pattern = re.compile(text_regex)

    limit = 100
    if max_deletions is not None and max_deletions < 20:
      limit = max_deletions

    cmd = 'conversations.history?channel=%s&limit=%i' % (channel_id, limit)
    if date_threshold is not None:
      if isinstance(date_threshold, datetime.datetime):
        pass  # Nothing has to be done
      elif isinstance(date_threshold, datetime.date):
        date_threshold = datetime.datetime.combine(date_threshold, datetime.time(0))
      elif not isinstance(date_threshold, datetime.datetime):
        raise ValueError('date_threshold must be either `datetime.date` or `datetime.datetime`!')
      cmd += '&latest=%f' % date_threshold.timestamp()

    try:
      deletions = 0
      msg_batch = cls.do_request(cmd, api_token)
      cmd += '&cursor=%s'
      while True:
        if len(msg_batch['messages']) == 0:
          break
        for m in msg_batch['messages']:
          if sender_pattern is not None and sender_pattern.search(m.get('username', '')) is None:
            print('Sender not matched')
            continue  # Skip deleting messages not coming from the specified user
          if pattern is not None and pattern.search(m.get('text', '')) is None:
            print('Pattern not matched')
            continue
          deletions += 1

          # Delete message
          response = cls.do_request('chat.delete', api_token, {
            'channel': channel_id,
            'ts': m['ts']
          })
          assert response['ok'], 'Non-OK response received: %s' % response
          print('Deleted message, response: %s' % response)

          if max_deletions is not None and deletions >= max_deletions:
            # Maximum number of deletions reached. Break
            print('Max_deletions (%i) reached!' % max_deletions)
            msg_batch['has_more'] = False  # do not continue
            break

        # Are there more messages to get?
        if msg_batch['has_more']:
          # Get next batch
          msg_batch = cls.do_request(cmd % msg_batch['response_metadata']['next_cursor'], api_token)
        else:
          break
      print("Done! Deleted %i messages" % deletions)
    except (URLError, HTTPError) as e:
      print('HTTP error! ' + e.read().decode(
        'utf-8'))  # Do not include a traceback, that's not interesting here (it's not a bug, but server complaining...)


if __name__ == '__main__':
  api = os.environ['SLACK_API_TOKEN']
  channel_id = SlackHandler.get_channel(api)
  text_regex = None  # 'h0_unet-att_1x1x3-adc'
  th = datetime.datetime.now().date()  # datetime.datetime.strptime('2019-11-25 08:20:00', '%Y-%m-%d %H:%M:%S')
  SlackHandler.cleanup_channel(api, channel_id, 'Attention-Gated-Networks', th, text_regex)
