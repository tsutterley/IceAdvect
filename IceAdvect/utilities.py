#!/usr/bin/env python
"""
utilities.py
Written by Tyler Sutterley (01/2026)
Download and management utilities for syncing files

PYTHON DEPENDENCIES:
    lxml: processing XML and HTML in Python
        https://pypi.python.org/pypi/lxml
    platformdirs: Python module for determining platform-specific directories
        https://pypi.org/project/platformdirs/

UPDATE HISTORY:
    Updated 01/2026: raise original exceptions in cases of HTTPError/URLError
    Written 01/2026
"""

from __future__ import print_function, division, annotations

import sys
import os
import re
import io
import ssl
import json
import shutil
import hashlib
import inspect
import logging
import pathlib
import warnings
import importlib
import posixpath
import subprocess
import lxml.etree
import platformdirs
import calendar, time

if sys.version_info[0] == 2:
    from urllib import quote_plus
    from cookielib import CookieJar
    from urlparse import urlparse
    import urllib2
else:
    from urllib.parse import quote_plus, urlparse
    from http.cookiejar import CookieJar
    import urllib.request as urllib2

__all__ = [
    "reify",
    "get_data_path",
    "get_cache_path",
    "import_dependency",
    "dependency_available",
    "is_valid_url",
    "Path",
    "URL",
    "compressuser",
    "get_hash",
    "get_git_revision_hash",
    "get_git_status",
    "url_split",
    "convert_arg_line_to_args",
    "get_unix_time",
    "_create_default_ssl_context",
    "_create_ssl_context_no_verify",
    "_set_ssl_context_options",
    "check_connection",
    "http_list",
    "from_http",
    "from_json",
]


class reify(object):
    """Class decorator that puts the result of the method it
    decorates into the instance"""

    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.__name__ = wrapped.__name__
        self.__doc__ = wrapped.__doc__

    def __get__(self, inst, objtype=None):
        if inst is None:
            return self
        val = self.wrapped(inst)
        setattr(inst, self.wrapped.__name__, val)
        return val


# PURPOSE: get absolute path within a package from a relative path
def get_data_path(relpath: list | str | pathlib.Path):
    """
    Get the absolute path within a package from a relative path

    Parameters
    ----------
    relpath: list, str or pathlib.Path
        relative path
    """
    # current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = pathlib.Path(filename).absolute().parent
    if isinstance(relpath, list):
        # use *splat operator to extract from list
        return filepath.joinpath(*relpath)
    elif isinstance(relpath, (str, pathlib.Path)):
        return filepath.joinpath(relpath)


# PURPOSE: get the path to the user cache directory
def get_cache_path(
    relpath: list | str | pathlib.Path | None = None, appname="advect"
):
    """
    Get the path to the user cache directory for an application

    Parameters
    ----------
    relpath: list, str, pathlib.Path or None
        relative path
    appname: str, default 'advect'
        application name
    """
    # get platform-specific cache directory
    filepath = platformdirs.user_cache_path(appname=appname, ensure_exists=True)
    if isinstance(relpath, list):
        # use *splat operator to extract from list
        filepath = filepath.joinpath(*relpath)
    elif isinstance(relpath, (str, pathlib.Path)):
        filepath = filepath.joinpath(relpath)
    return pathlib.Path(filepath)


def import_dependency(
    name: str, extra: str = "", raise_exception: bool = False
):
    """
    Import an optional dependency

    Adapted from ``pandas.compat._optional::import_optional_dependency``

    Parameters
    ----------
    name: str
        Module name
    extra: str, default ""
        Additional text to include in the ``ImportError`` message
    raise_exception: bool, default False
        Raise an ``ImportError`` if the module is not found

    Returns
    -------
    module: obj
        Imported module
    """
    # check if the module name is a string
    msg = f"Invalid module name: '{name}'; must be a string"
    assert isinstance(name, str), msg
    # default error if module cannot be imported
    err = f"Missing optional dependency '{name}'. {extra}"
    module = type("module", (), {})
    # try to import the module
    try:
        module = importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        if raise_exception:
            raise ImportError(err) from exc
        else:
            logging.debug(err)
    # return the module
    return module


def dependency_available(name: str, minversion: str | None = None):
    """
    Checks whether a module is installed without importing it

    Adapted from ``xarray.namedarray.utils.module_available``

    Parameters
    ----------
    name: str
        Module name
    minversion : str, optional
        Minimum version of the module

    Returns
    -------
    available : bool
        Whether the module is installed
    """
    # check if module is available
    if importlib.util.find_spec(name) is None:
        return False
    # check if the version is greater than the minimum required
    if minversion is not None:
        version = importlib.metadata.version(name)
        return version >= minversion
    # return if both checks are passed
    return True


def is_valid_url(url: str) -> bool:
    """
    Checks if a string is a valid URL

    Parameters
    ----------
    url: str
        URL to check
    """
    try:
        result = urlparse(str(url))
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False


def Path(filename: str | pathlib.Path, *args, **kwargs):
    """
    Create a ``URL`` or ``pathlib.Path`` object

    Parameters
    ----------
    filename: str or pathlib.Path
        file path or URL
    """
    if is_valid_url(filename):
        return URL(filename, *args, **kwargs)
    else:
        return pathlib.Path(filename, *args, **kwargs).expanduser()


class URL:
    """Handles URLs similar to ``pathlib.Path`` objects"""

    def __init__(self, urlname: str | pathlib.Path, *args, **kwargs):
        """Initialize a ``URL`` object"""
        self.urlname = str(urlname)
        self._raw_paths = list(url_split(self.urlname))
        self._headers = {}

    @classmethod
    def from_parts(cls, parts: str | list | tuple):
        """
        Return a ``URL`` object from components

        Parameters
        ----------
        parts: str, list or tuple
            URL components
        """
        # verify that parts are iterable as list or tuple
        if isinstance(parts, str):
            return cls(parts)
        else:
            return cls("/".join([*parts]))

    def joinpath(self, *pathsegments: list[str]):
        """Append URL components to existing

        Parameters
        ----------
        pathsegments: list[str]
            URL components to append
        """
        return URL("/".join([*self._raw_paths, *pathsegments]))

    def resolve(self):
        """Resolve the URL"""
        return URL("/".join([*self._raw_paths]))

    def is_file(self):
        """Boolean flag if path is a local file"""
        return False

    def is_dir(self):
        """Boolean flag if path is a local directory"""
        return False

    def geturl(self):
        """String representation of the ``URL`` object"""
        return self._components.geturl()

    def get(self, *args, **kwargs):
        """Get contents from URL"""
        return from_http(self.urlname, headers=self._headers, *args, **kwargs)

    def headers(self, *args, **kwargs):
        """Get headers from URL"""
        self.urlopen(*args, **kwargs)
        return self._headers

    def load(self, *args, **kwargs):
        """Load JSON response from URL"""
        return from_json(self.urlname, headers=self._headers, *args, **kwargs)

    def ping(self, *args, **kwargs) -> bool:
        """Ping URL to check connection"""
        return check_connection(self.urlname, *args, **kwargs)

    def read(self, *args, **kwargs):
        """Open URL and read response"""
        return self.urlopen(*args, **kwargs).read()

    def urlopen(self, *args, **kwargs):
        """Open URL and return response"""
        request = urllib2.Request(self.urlname)
        response = urllib2.urlopen(request, *args, **kwargs)
        self._headers.update(
            {k.lower(): v for k, v in response.headers.items()}
        )
        return response

    @property
    def name(self):
        """URL basename"""
        return pathlib.PurePosixPath(self.urlname).name

    @property
    def netloc(self):
        """URL network location"""
        return self._components.netloc

    @property
    def parent(self):
        """URL parent path as a ``URL`` object"""
        paths = url_split(self.urlname)[:-1]
        return URL.from_parts(paths)

    @property
    def parents(self):
        """URL parents as a list of ``URL`` objects"""
        paths = url_split(self.urlname)
        return [URL.from_parts(paths[:i]) for i in range(len(paths) - 1, 0, -1)]

    @property
    def parts(self):
        """URL parts as a tuple"""
        paths = url_split(self._components.path)
        return (self.scheme, self.netloc, *paths)

    @property
    def scheme(self):
        """URL scheme"""
        return self._components.scheme + "://"

    @property
    def stem(self):
        """URL stem"""
        return pathlib.PurePosixPath(self.urlname).stem

    @property
    def _components(self):
        """
        URL parsed into six components using ``urlparse``
        """
        return urlparse(self.urlname)

    def __repr__(self):
        """Representation of the ``URL`` object"""
        return str(self.urlname)

    def __str__(self):
        """String representation of the ``URL`` object"""
        return str(self.urlname)

    def __div__(self, other):
        """Join URL components using the division operator"""
        return self.joinpath(other)

    def __truediv__(self, other):
        """Join URL components using the division operator"""
        return self.joinpath(other)


def compressuser(filename: str | pathlib.Path):
    """
    Tilde-compress a file to be relative to the home directory

    Parameters
    ----------
    filename: str or pathlib.Path
        input filename to compress
    """
    # attempt to compress filename relative to home directory
    filename = pathlib.Path(filename).expanduser().absolute()
    try:
        relative_to = filename.relative_to(pathlib.Path().home())
    except (ValueError, AttributeError) as exc:
        return filename
    else:
        return pathlib.Path("~").joinpath(relative_to)


# PURPOSE: get the hash value of a file
def get_hash(local: str | io.IOBase | pathlib.Path, algorithm: str = "md5"):
    """
    Get the hash value from a local file or ``BytesIO`` object

    Parameters
    ----------
    local: obj, str or pathlib.Path
        BytesIO object or path to file
    algorithm: str, default 'md5'
        hashing algorithm for checksum validation
    """
    # check if open file object or if local file exists
    if isinstance(local, io.IOBase):
        # generate checksum hash for a given type
        if algorithm in hashlib.algorithms_available:
            return hashlib.new(algorithm, local.getvalue()).hexdigest()
        else:
            raise ValueError(f"Invalid hashing algorithm: {algorithm}")
    elif isinstance(local, (str, pathlib.Path)):
        # generate checksum hash for local file
        local = pathlib.Path(local).expanduser()
        # if file currently doesn't exist, return empty string
        if not local.exists():
            return ""
        # open the local_file in binary read mode
        with local.open(mode="rb") as local_buffer:
            # generate checksum hash for a given type
            if algorithm in hashlib.algorithms_available:
                return hashlib.new(algorithm, local_buffer.read()).hexdigest()
            else:
                raise ValueError(f"Invalid hashing algorithm: {algorithm}")
    else:
        return ""


# PURPOSE: get the git hash value
def get_git_revision_hash(refname: str = "HEAD", short: bool = False):
    """
    Get the ``git`` hash value for a particular reference

    Parameters
    ----------
    refname: str, default HEAD
        Symbolic reference name
    short: bool, default False
        Return the shorted hash value
    """
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = pathlib.Path(filename).absolute().parent.parent
    gitpath = basepath.joinpath(".git")
    # build command
    cmd = ["git", f"--git-dir={gitpath}", "rev-parse"]
    cmd.append("--short") if short else None
    cmd.append(refname)
    # get output
    with warnings.catch_warnings():
        return str(subprocess.check_output(cmd), encoding="utf8").strip()


# PURPOSE: get the current git status
def get_git_status():
    """Get the status of a ``git`` repository as a boolean value"""
    # get path to .git directory from current file path
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basepath = pathlib.Path(filename).absolute().parent.parent
    gitpath = basepath.joinpath(".git")
    # build command
    cmd = ["git", f"--git-dir={gitpath}", "status", "--porcelain"]
    with warnings.catch_warnings():
        return bool(subprocess.check_output(cmd))


# PURPOSE: recursively split a url path
def url_split(s: str):
    """
    Recursively split a url path into a list

    Parameters
    ----------
    s: str
        url string
    """
    head, tail = posixpath.split(str(s))
    if head in ("http:", "https:", "ftp:", "s3:"):
        return (s,)
    elif head in ("", posixpath.sep):
        return (tail,)
    return url_split(head) + (tail,)


# PURPOSE: convert file lines to arguments
def convert_arg_line_to_args(arg_line):
    """
    Convert file lines to arguments

    Parameters
    ----------
    arg_line: str
        line string containing a single argument and/or comments
    """
    # remove commented lines and after argument comments
    for arg in re.sub(r"\#(.*?)$", r"", arg_line).split():
        if not arg.strip():
            continue
        yield arg


# PURPOSE: returns the Unix timestamp value for a formatted date string
def get_unix_time(time_string: str, format: str = "%Y-%m-%d %H:%M:%S"):
    """
    Get the Unix timestamp value for a formatted date string

    Parameters
    ----------
    time_string: str
        formatted time string to parse
    format: str, default '%Y-%m-%d %H:%M:%S'
        format for input time string
    """
    try:
        parsed_time = time.strptime(time_string.rstrip(), format)
    except (TypeError, ValueError):
        pass
    else:
        return calendar.timegm(parsed_time)


def _create_default_ssl_context() -> ssl.SSLContext:
    """Creates the default SSL context"""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _set_ssl_context_options(context)
    context.options |= ssl.OP_NO_COMPRESSION
    return context


def _create_ssl_context_no_verify() -> ssl.SSLContext:
    """Creates an SSL context for unverified connections"""
    context = _create_default_ssl_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _set_ssl_context_options(context: ssl.SSLContext) -> None:
    """Sets the default options for the SSL context"""
    if sys.version_info >= (3, 10) or ssl.OPENSSL_VERSION_INFO >= (1, 1, 0, 7):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
    else:
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1


# default ssl context
_default_ssl_context = _create_ssl_context_no_verify()


# PURPOSE: check connection with http host
def check_connection(
    HOST: str,
    context: ssl.SSLContext = _default_ssl_context,
    timeout: int = 20,
):
    """
    Check internet connection with http host

    Parameters
    ----------
    HOST: str
        remote http host
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    timeout: int, default 20
        timeout in seconds for blocking operations
    """
    # attempt to connect to http host
    try:
        urllib2.urlopen(HOST, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        exc.message = "Check internet connection"
        raise
    else:
        return True


# PURPOSE: list a directory on an Apache http Server
def http_list(
    HOST: str | list,
    timeout: int | None = None,
    context: ssl.SSLContext = _default_ssl_context,
    parser=lxml.etree.HTMLParser(),
    format: str = "%Y-%m-%d %H:%M",
    pattern: str = "",
    sort: bool = False,
    **kwargs,
):
    """
    List a directory on an Apache http Server

    Parameters
    ----------
    HOST: str or list
        remote http host path
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    parser: obj, default lxml.etree.HTMLParser()
        HTML parser for ``lxml``
    format: str, default '%Y-%m-%d %H:%M'
        format for input time string
    pattern: str, default ''
        regular expression pattern for reducing list
    sort: bool, default False
        sort output list

    Returns
    -------
    colnames: list
        column names in a directory
    collastmod: list
        last modification times for items in the directory
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try listing from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST), **kwargs)
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        exc.message = "Check internet connection"
        raise
    else:
        # read and parse request for files (column names and modified times)
        tree = lxml.etree.parse(response, parser)
        colnames = tree.xpath("//tr/td[not(@*)]//a/@href")
        # get the Unix timestamp value for a modification time
        collastmod = [
            get_unix_time(i, format=format)
            for i in tree.xpath('//tr/td[@align="right"][1]/text()')
        ]
        # reduce using regular expression pattern
        if pattern:
            i = [i for i, f in enumerate(colnames) if re.search(pattern, f)]
            # reduce list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        # sort the list
        if sort:
            i = [i for i, j in sorted(enumerate(colnames), key=lambda i: i[1])]
            # sort list of column names and last modified times
            colnames = [colnames[indice] for indice in i]
            collastmod = [collastmod[indice] for indice in i]
        # return the list of column names and last modified times
        return (colnames, collastmod)


# PURPOSE: download a file from a http host
def from_http(
    HOST: str | list,
    timeout: int | None = None,
    context: ssl.SSLContext = _default_ssl_context,
    local: str | pathlib.Path | None = None,
    hash: str = "",
    chunk: int = 16384,
    headers: dict = {},
    verbose: bool = False,
    fid=sys.stdout,
    mode: oct = 0o775,
    **kwargs,
):
    """
    Download a file from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    local: str, pathlib.Path or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    headers: dict, default {}
        dictionary of headers to append from url request
    verbose: bool, default False
        print file transfer information
    fid: obj, default sys.stdout
        open file object to print if verbose
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    """
    # create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(stream=fid, level=loglevel)
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try downloading from http
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST), **kwargs)
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        exc.message = "Check internet connection"
        raise
    else:
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        # save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
        # copy headers from response
        headers.update({k.lower(): v for k, v in response.getheaders()})
        # generate checksum hash for remote file
        remote_hash = hashlib.md5(remote_buffer.getvalue()).hexdigest()
        # compare checksums
        if local and (hash != remote_hash):
            # convert to absolute path
            local = pathlib.Path(local).expanduser().absolute()
            # create directory if non-existent
            local.parent.mkdir(mode=mode, parents=True, exist_ok=True)
            # print file information
            args = (posixpath.join(*HOST), str(local))
            logging.info("{0} -->\n\t{1}".format(*args))
            # store bytes to file using chunked transfer encoding
            remote_buffer.seek(0)
            with local.open(mode="wb") as f:
                shutil.copyfileobj(remote_buffer, f, chunk)
            # change the permissions mode
            local.chmod(mode)
        # return the bytesIO object
        remote_buffer.seek(0)
        return remote_buffer


# PURPOSE: load a JSON response from a http host
def from_json(
    HOST: str | list,
    timeout: int | None = None,
    context: ssl.SSLContext = _default_ssl_context,
    headers: dict = {},
) -> dict:
    """
    Load a JSON response from a http host

    Parameters
    ----------
    HOST: str or list
        remote http host path split as list
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    headers: dict, default {}
        dictionary of headers to append from url request
    """
    # verify inputs for remote http host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try loading JSON from http
    try:
        # Create and submit request for JSON response
        request = urllib2.Request(posixpath.join(*HOST))
        request.add_header("Accept", "application/json")
        response = urllib2.urlopen(request, timeout=timeout, context=context)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        exc.message = "Check internet connection"
        raise
    else:
        # copy headers from response
        headers.update({k.lower(): v for k, v in response.getheaders()})
        # load JSON response
        return json.loads(response.read())
