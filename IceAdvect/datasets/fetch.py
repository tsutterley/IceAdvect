#!/usr/bin/env python
"""
fetch.py
Written by Tyler Sutterley (01/2026)
Download routines for NASA Earthdata files

UPDATE HISTORY:
    Updated 01/2026: return the list of queried granules from fetch
    Written 01/2026
"""

from __future__ import print_function, division, annotations

import io
import os
import re
import ssl
import sys
import json
import netrc
import base64
import shutil
import getpass
import hashlib
import logging
import pathlib
import builtins
import posixpath
from IceAdvect.utilities import (
    CookieJar,
    urllib2,
    _default_ssl_context,
    get_cache_path,
    url_split,
)

__all__ = [
    "s3_client",
    "s3_filesystem",
    "s3_bucket",
    "s3_key",
    "s3_presigned_url",
    "generate_presigned_url",
    "attempt_login",
    "build_opener",
    "get_token",
    "list_tokens",
    "revoke_token",
    "check_credentials",
    "from_earthdata",
    "cmr_filter_json",
    "cmr",
    "fetch",
]


# NASA on-prem DAAC providers
_daac_providers = {
    "gesdisc": "GES_DISC",
    "ghrcdaac": "GHRC_DAAC",
    "lpdaac": "LPDAAC_ECS",
    "nsidc": "NSIDC_ECS",
    "ornldaac": "ORNL_DAAC",
    "podaac": "PODAAC",
}

# NASA Cumulus AWS providers
_s3_providers = {
    "gesdisc": "GES_DISC",
    "ghrcdaac": "GHRC_DAAC",
    "lpdaac": "LPCLOUD",
    "nsidc": "NSIDC_CPRD",
    "ornldaac": "ORNL_CLOUD",
    "podaac": "POCLOUD",
}

# NASA Cumulus AWS S3 credential endpoints
_s3_endpoints = {
    "gesdisc": "https://data.gesdisc.earthdata.nasa.gov/s3credentials",
    "ghrcdaac": "https://data.ghrc.earthdata.nasa.gov/s3credentials",
    "lpdaac": "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
    "nsidc": "https://data.nsidc.earthdatacloud.nasa.gov/s3credentials",
    "ornldaac": "https://data.ornldaac.earthdata.nasa.gov/s3credentials",
    "podaac": "https://archive.podaac.earthdata.nasa.gov/s3credentials",
}

# NASA Cumulus AWS S3 buckets
_s3_buckets = {
    "gesdisc": "gesdisc-cumulus-prod-protected",
    "ghrcdaac": "ghrc-cumulus-dev",
    "lpdaac": "lp-prod-protected",
    "nsidc": "nsidc-cumulus-prod-protected",
    "ornldaac": "ornl-cumulus-prod-protected",
    "podaac": "podaac-ops-cumulus-protected",
}


# PURPOSE: get AWS s3 client for NSIDC Cumulus
def s3_client(
    HOST: str = _s3_endpoints["nsidc"],
    timeout: int | None = None,
    region_name: str = "us-west-2",
):
    """
    Get AWS s3 client for NSIDC data in the cloud
    https://data.nsidc.earthdatacloud.nasa.gov/s3credentials

    Parameters
    ----------
    HOST: str
        NSIDC AWS S3 credential host
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    region_name: str, default 'us-west-2'
        AWS region name

    Returns
    -------
    client: obj
        AWS s3 client for NSIDC Cumulus
    """
    import boto3

    request = urllib2.Request(HOST)
    response = urllib2.urlopen(request, timeout=timeout)
    cumulus = json.loads(response.read())
    # get AWS client object
    client = boto3.client(
        "s3",
        aws_access_key_id=cumulus["accessKeyId"],
        aws_secret_access_key=cumulus["secretAccessKey"],
        aws_session_token=cumulus["sessionToken"],
        region_name=region_name,
    )
    # return the AWS client for region
    return client


# PURPOSE: get AWS s3 file system for NSIDC Cumulus
def s3_filesystem(
    HOST: str = _s3_endpoints["nsidc"],
    timeout: int | None = None,
    region_name: str = "us-west-2",
):
    """
    Get AWS s3 file system object for NSIDC data in the cloud
    https://data.nsidc.earthdatacloud.nasa.gov/s3credentials

    Parameters
    ----------
    HOST: str
        NSIDC AWS S3 credential host
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    region_name: str, default 'us-west-2'
        AWS region name

    Returns
    -------
    session: obj
        AWS s3 file system session for NSIDC Cumulus
    """
    import s3fs

    request = urllib2.Request(HOST)
    response = urllib2.urlopen(request, timeout=timeout)
    cumulus = json.loads(response.read())
    # get AWS file system session object
    session = s3fs.S3FileSystem(
        anon=False,
        key=cumulus["accessKeyId"],
        secret=cumulus["secretAccessKey"],
        token=cumulus["sessionToken"],
        client_kwargs=dict(region_name=region_name),
    )
    # return the AWS session for region
    return session


# PURPOSE: get a s3 bucket name from a presigned url
def s3_bucket(presigned_url: str):
    """
    Get a s3 bucket name from a presigned url

    Parameters
    ----------
    presigned_url: str
        s3 presigned url

    Returns
    -------
    bucket: str
        s3 bucket name
    """
    host = url_split(presigned_url)
    bucket = re.sub(r"s3:\/\/", r"", host[0], re.IGNORECASE)
    return bucket


# PURPOSE: get a s3 bucket key from a presigned url
def s3_key(presigned_url: str):
    """
    Get a s3 bucket key from a presigned url

    Parameters
    ----------
    presigned_url: str
        s3 presigned url or https url

    Returns
    -------
    key: str
        s3 bucket key for object
    """
    host = url_split(presigned_url)
    # check if url is https url or s3 presigned url
    if presigned_url.startswith("http"):
        # use NSIDC format for s3 keys from https
        parsed = [p for part in host[-4:-1] for p in part.split(".")]
        # join parsed url parts to form bucket key
        key = posixpath.join(*parsed, host[-1])
    else:
        # join presigned url to form bucket key
        key = posixpath.join(*host[1:])
    # return the s3 bucket key for object
    return key


# PURPOSE: get a s3 presigned url from a bucket and key
def s3_presigned_url(bucket: str, key: str):
    """
    Get a s3 presigned url from a bucket and object key

    Parameters
    ----------
    bucket: str
        s3 bucket name
    key: str
        s3 bucket key for object

    Returns
    -------
    presigned_url: str
        s3 presigned url
    """
    return posixpath.join("s3://", bucket, key)


# PURPOSE: generate a s3 presigned https url from a bucket and key
def generate_presigned_url(bucket: str, key: str, expiration: int = 3600):
    """
    Generate a presigned https URL to share an S3 object

    Parameters
    ----------
    bucket: str
        s3 bucket name
    key: str
        s3 bucket key for object
    expiration: int
        Time in seconds for the presigned URL to remain valid

    Returns
    -------
    presigned_url: str
        s3 presigned https url
    """
    import boto3

    # generate a presigned URL for S3 object
    s3 = boto3.client("s3")
    try:
        response = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expiration,
        )
    except Exception as exc:
        logging.error(exc)
        return None
    # The response contains the presigned URL
    return response


# PURPOSE: attempt to build an opener with netrc
def attempt_login(
    urs: str = "urs.earthdata.nasa.gov",
    context: ssl.SSLContext = _default_ssl_context,
    password_manager: bool = False,
    get_ca_certs: bool = False,
    redirect: bool = False,
    authorization_header: bool = True,
    **kwargs,
):
    """
    attempt to build a ``urllib`` opener for NASA Earthdata

    Parameters
    ----------
    urs: str, default urs.earthdata.nasa.gov
        Earthdata login URS 3 host
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    password_manager: bool, default True
        Create password manager context using default realm
    get_ca_certs: bool, default False
        Get list of loaded “certification authority” certificates
    redirect: bool, default False
        Create redirect handler object
    authorization_header: bool, default False
        Add base64 encoded authorization header to opener
    username: str, default from environmental variable
        NASA Earthdata username
    password: str, default from environmental variable
        NASA Earthdata password
    retries: int, default 5
        number of retry attempts
    netrc: str, default ~/.netrc
        path to .netrc file for authentication

    Returns
    -------
    opener: obj
        OpenerDirector instance
    """
    # set default keyword arguments
    kwargs.setdefault("username", os.environ.get("EARTHDATA_USERNAME"))
    kwargs.setdefault("password", os.environ.get("EARTHDATA_PASSWORD"))
    kwargs.setdefault("retries", 5)
    kwargs.setdefault("netrc", pathlib.Path.home().joinpath(".netrc"))
    try:
        # only necessary on jupyterhub
        kwargs["netrc"].chmod(mode=0o600)
        # try retrieving credentials from netrc
        username, _, password = netrc.netrc(kwargs["netrc"]).authenticators(urs)
    except Exception as exc:
        logging.error(exc)
        # try retrieving credentials from environmental variables
        username, password = (kwargs["username"], kwargs["password"])
    # if username or password are not available
    if not username:
        username = builtins.input(f"Username for {urs}: ")
    if not password:
        password = getpass.getpass(prompt=f"Password for {username}@{urs}: ")
    # for each retry
    for retry in range(kwargs["retries"]):
        # build an opener for urs with credentials
        opener = build_opener(
            username,
            password,
            context=context,
            password_manager=password_manager,
            get_ca_certs=get_ca_certs,
            redirect=redirect,
            authorization_header=authorization_header,
            urs=urs,
        )
        # try logging in by check credentials
        try:
            check_credentials()
        except Exception as exc:
            logging.error(exc)
        else:
            return opener
        # reattempt login
        username = builtins.input(f"Username for {urs}: ")
        password = getpass.getpass(prompt=f"Password for {username}@{urs}: ")
    # reached end of available retries
    raise RuntimeError("End of Retries: Check NASA Earthdata credentials")


# PURPOSE: "login" to NASA Earthdata with supplied credentials
def build_opener(
    username: str,
    password: str,
    context: ssl.SSLContext = _default_ssl_context,
    password_manager: bool = True,
    get_ca_certs: bool = False,
    redirect: bool = False,
    authorization_header: bool = False,
    urs: str = "https://urs.earthdata.nasa.gov",
):
    """
    Build ``urllib`` opener for NASA Earthdata with supplied credentials

    Parameters
    ----------
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    password_manager: bool, default True
        Create password manager context using default realm
    get_ca_certs: bool, default False
        Get list of loaded “certification authority” certificates
    redirect: bool, default False
        Create redirect handler object
    authorization_header: bool, default False
        Add base64 encoded authorization header to opener
    urs: str, default 'https://urs.earthdata.nasa.gov'
        Earthdata login URS 3 host

    Returns
    -------
    opener: obj
        ``OpenerDirector`` instance
    """
    # https://docs.python.org/3/howto/urllib2.html#id5
    handler = []
    # create a password manager
    if password_manager:
        password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
        # Add the username and password for NASA Earthdata Login system
        password_mgr.add_password(None, urs, username, password)
        handler.append(urllib2.HTTPBasicAuthHandler(password_mgr))
    # Create cookie jar for storing cookies. This is used to store and return
    # the session cookie given to use by the data server (otherwise will just
    # keep sending us back to Earthdata Login to authenticate).
    cookie_jar = CookieJar()
    handler.append(urllib2.HTTPCookieProcessor(cookie_jar))
    # SSL context handler
    if get_ca_certs:
        context.get_ca_certs()
    handler.append(urllib2.HTTPSHandler(context=context))
    # redirect handler
    if redirect:
        handler.append(urllib2.HTTPRedirectHandler())
    # create "opener" (OpenerDirector instance)
    opener = urllib2.build_opener(*handler)
    # Encode username/password for request authorization headers
    # add Authorization header to opener
    if authorization_header:
        b64 = base64.b64encode(f"{username}:{password}".encode())
        opener.addheaders = [("Authorization", f"Basic {b64.decode()}")]
    # Now all calls to urllib2.urlopen use our opener.
    urllib2.install_opener(opener)
    # All calls to urllib2.urlopen will now use handler
    # Make sure not to include the protocol in with the URL, or
    # HTTPPasswordMgrWithDefaultRealm will be confused.
    return opener


# PURPOSE: generate a NASA Earthdata user token
def get_token(
    HOST: str = "https://urs.earthdata.nasa.gov/api/users/token",
    build: bool = True,
    urs: str = "urs.earthdata.nasa.gov",
    **kwargs,
):
    """
    Generate a NASA Earthdata User Token

    Parameters
    ----------
    HOST: str or list
        NASA Earthdata token API host
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    build: bool, default True
        Build opener and check credentials
    timeout: int or NoneType, default Nonedata'
        timeout in seconds for blocking operations
    urs: str, default 'urs.earthdata.nasa.gov'
        NASA Earthdata URS 3 host

    Returns
    -------
    token: dict
        JSON response with NASA Earthdata User Token
    """
    # set default keyword arguments
    kwargs.setdefault("username", os.environ.get("EARTHDATA_USERNAME"))
    kwargs.setdefault("password", os.environ.get("EARTHDATA_PASSWORD"))
    # attempt to build urllib2 opener and check credentials
    if build:
        attempt_login(
            urs, password_manager=False, authorization_header=True, **kwargs
        )
    # create post response with Earthdata token API
    try:
        request = urllib2.Request(HOST, method="POST")
        response = urllib2.urlopen(request)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        raise RuntimeError("Check internet connection") from exc
    # read and return JSON response
    return json.loads(response.read())


# PURPOSE: generate a NASA Earthdata user token
def list_tokens(
    HOST: str = "https://urs.earthdata.nasa.gov/api/users/tokens",
    build: bool = True,
    urs: str = "urs.earthdata.nasa.gov",
    **kwargs,
):
    """
    List the current associated NASA Earthdata User Tokens

    Parameters
    ----------
    HOST: str
        NASA Earthdata list token API host
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    build: bool, default True
        Build opener and check credentials
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    urs: str, default 'urs.earthdata.nasa.gov'
        NASA Earthdata URS 3 host

    Returns
    -------
    tokens: list
        JSON response with NASA Earthdata User Tokens
    """
    # set default keyword arguments
    kwargs.setdefault("username", os.environ.get("EARTHDATA_USERNAME"))
    kwargs.setdefault("password", os.environ.get("EARTHDATA_PASSWORD"))
    # attempt to build urllib2 opener and check credentials
    if build:
        attempt_login(
            urs,
            password_manager=False,
            authorization_header=True,
            **kwargs,
        )
    # create get response with Earthdata list tokens API
    try:
        request = urllib2.Request(HOST)
        response = urllib2.urlopen(request)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        raise RuntimeError("Check internet connection") from exc
    # read and return JSON response
    return json.loads(response.read())


# PURPOSE: revoke a NASA Earthdata user token
def revoke_token(
    token: str,
    HOST: str = f"https://urs.earthdata.nasa.gov/api/users/revoke_token",
    build: bool = True,
    urs: str = "urs.earthdata.nasa.gov",
    **kwargs,
):
    """
    Generate a NASA Earthdata User Token

    Parameters
    ----------
    token: str
        NASA Earthdata token to be revoked
    HOST: str
        NASA Earthdata revoke token API host
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    build: bool, default True
        Build opener and check credentials
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    urs: str, default 'urs.earthdata.nasa.gov'
        NASA Earthdata URS 3 host
    """
    # set default keyword arguments
    kwargs.setdefault("username", os.environ.get("EARTHDATA_USERNAME"))
    kwargs.setdefault("password", os.environ.get("EARTHDATA_PASSWORD"))
    # attempt to build urllib2 opener and check credentials
    if build:
        attempt_login(
            urs,
            password_manager=False,
            authorization_header=True,
            **kwargs,
        )
    # full path for NASA Earthdata revoke token API
    url = f"{HOST}?token={token}"
    # create post response with Earthdata revoke tokens API
    try:
        request = urllib2.Request(url, method="POST")
        response = urllib2.urlopen(request)
    except urllib2.HTTPError as exc:
        logging.debug(exc.code)
        raise RuntimeError(exc.reason) from exc
    except urllib2.URLError as exc:
        logging.debug(exc.reason)
        raise RuntimeError("Check internet connection") from exc
    # verbose response
    logging.debug(f"Token Revoked: {token}")


# PURPOSE: check that entered NASA Earthdata credentials are valid
def check_credentials():
    """
    Check that entered NASA Earthdata credentials are valid
    """
    try:
        remote_path = "https://urs.earthdata.nasa.gov/api/users/tokens"
        request = urllib2.Request(url=remote_path)
        urllib2.urlopen(request, timeout=20)
    except urllib2.HTTPError as exc:
        raise RuntimeError("Check your NASA Earthdata credentials") from exc
    except urllib2.URLError as exc:
        raise RuntimeError("Check internet connection") from exc
    else:
        return True


# PURPOSE: download a file from a NASA Earthdata provider
def from_earthdata(
    HOST: str | list,
    build: bool = True,
    timeout: int | None = None,
    urs: str = "urs.earthdata.nasa.gov",
    local: str | pathlib.Path | None = None,
    hash: str = "",
    chunk: int = 16384,
    verbose: bool = False,
    mode: oct = 0o775,
    **kwargs,
):
    """
    Download a file from a NASA Earthdata provider

    Parameters
    ----------
    HOST: str or list
        remote https host
    username: str or NoneType, default None
        NASA Earthdata username
    password: str or NoneType, default None
        NASA Earthdata password
    build: bool, default True
        Build opener and check credentials
    timeout: int or NoneType, default None
        timeout in seconds for blocking operations
    urs: str, default 'urs.earthdata.nasa.gov'
        NASA Earthdata URS 3 host
    local: str or NoneType, default None
        path to local file
    hash: str, default ''
        MD5 hash of local file
    chunk: int, default 16384
        chunk size for transfer encoding
    verbose: bool, default False
        print file transfer information
    mode: oct, default 0o775
        permissions mode of output local file

    Returns
    -------
    remote_buffer: obj
        BytesIO representation of file
    response_error: str or None
        notification for response error
    """
    # set default keyword arguments
    kwargs.setdefault("username", os.environ.get("EARTHDATA_USERNAME"))
    kwargs.setdefault("password", os.environ.get("EARTHDATA_PASSWORD"))
    # create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)
    # attempt to build urllib2 opener and check credentials
    if build:
        attempt_login(urs, **kwargs)
    # verify inputs for remote https host
    if isinstance(HOST, str):
        HOST = url_split(HOST)
    # try downloading from https
    try:
        # Create and submit request.
        request = urllib2.Request(posixpath.join(*HOST))
        response = urllib2.urlopen(request, timeout=timeout)
    except (urllib2.HTTPError, urllib2.URLError) as exc:
        logging.error(exc)
        response_error = "Download error from {0}".format(posixpath.join(*HOST))
        return (False, response_error)
    else:
        # copy remote file contents to bytesIO object
        remote_buffer = io.BytesIO()
        shutil.copyfileobj(response, remote_buffer, chunk)
        remote_buffer.seek(0)
        # save file basename with bytesIO object
        remote_buffer.filename = HOST[-1]
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
            local.chmod(mode=mode)
        # return the bytesIO object
        remote_buffer.seek(0)
        return (remote_buffer, None)


# PURPOSE: filter the CMR json response for desired data files
def cmr_filter_json(
    search_results: dict,
    endpoint: str = "data",
):
    """
    Filter the CMR json response for desired data files

    Parameters
    ----------
    search_results: dict
        json response from CMR query
    endpoint: str, default 'data'
        url endpoint type

            - ``'data'``: NASA Earthdata https archive
            - ``'opendap'``: NASA Earthdata OPeNDAP archive
            - ``'s3'``: NASA Earthdata Cumulus AWS S3 bucket

    Returns
    -------
    granule_urls: list
        granule urls from NSIDC
    """
    # output list of granule urls
    granule_urls = []
    # check that there are urls for request
    if ("feed" not in search_results) or (
        "entry" not in search_results["feed"]
    ):
        return granule_urls
    # descriptor links for each endpoint
    rel = {}
    rel["data"] = "http://esipfed.org/ns/fedsearch/1.1/data#"
    rel["opendap"] = "http://esipfed.org/ns/fedsearch/1.1/service#"
    rel["s3"] = "http://esipfed.org/ns/fedsearch/1.1/s3#"
    # iterate over references and get cmr location
    for entry in search_results["feed"]["entry"]:
        for link in entry["links"]:
            # skip links without descriptors
            if "rel" not in link.keys():
                continue
            if "inherited" in link.keys():
                continue
            # append if selected endpoint and request type
            if link["rel"] == rel[endpoint]:
                granule_urls.append(link["href"])
    # return the list of urls and granule ids
    return granule_urls


# PURPOSE: cmr queries
def cmr(
    collection_concept_id: str,
    producer_granule_id: str | None = None,
    readable_granule_name: list | None = None,
    provider: str = "NSIDC_CPRD",
    endpoint: str = "data",
    opener=None,
    context: ssl.SSLContext = _default_ssl_context,
    verbose: bool = False,
    **kwargs,
):
    """
    Query the NASA Common Metadata Repository (CMR)

    Parameters
    ----------
    collection_concept_id: str
        Earthdata Collection ID of the data product
    producer_granule_id: str or NoneType, default None
        CMR producer granule id
    readable_granule_name: list or NoneType, default None
        list of CMR readable granule names
    provider: str, default 'NSIDC_CPRD'
        CMR data provider
    endpoint: str, default 'data'
        url endpoint type

            - ``'data'``: NASA Earthdata https archive
            - ``'opendap'``: NASA Earthdata OPeNDAP archive
            - ``'s3'``: NASA Earthdata Cumulus AWS S3 bucket
    opener: obj or NoneType, default None
        ``OpenerDirector`` instance
    context: obj, default IceAdvect.utilities._default_ssl_context
        SSL context for ``urllib`` opener object
    verbose: bool, default False
        print file transfer information

    Returns
    -------
    granule_urls: list
        granule urls
    """
    # create logger
    loglevel = logging.INFO if verbose else logging.CRITICAL
    logging.basicConfig(level=loglevel)
    # attempt to build urllib2 opener
    if opener is None:
        # build urllib2 opener with SSL context
        # https://docs.python.org/3/howto/urllib2.html#id5
        handler = []
        # Create cookie jar for storing cookies
        cookie_jar = CookieJar()
        handler.append(urllib2.HTTPCookieProcessor(cookie_jar))
        handler.append(urllib2.HTTPSHandler(context=context))
        # create "opener" (OpenerDirector instance)
        opener = urllib2.build_opener(*handler)
    # build CMR query
    cmr_query_type = "granules"
    cmr_format = "json"
    cmr_page_size = 2000
    CMR_HOST = [
        "https://cmr.earthdata.nasa.gov",
        "search",
        f"{cmr_query_type}.{cmr_format}",
    ]
    # build list of CMR query parameters
    CMR_KEYS = []
    CMR_KEYS.append(f"?provider={provider}")
    CMR_KEYS.append("&sort_key[]=start_date")
    CMR_KEYS.append("&sort_key[]=producer_granule_id")
    CMR_KEYS.append(f"&page_size={cmr_page_size}")
    # append collection concept ID string
    CMR_KEYS.append(f"&collection-concept-id={collection_concept_id}")
    # append producer granule id string
    if producer_granule_id is not None:
        CMR_KEYS.append(f"&producer-granule-id={producer_granule_id}")
    # append readable granule name strings
    if readable_granule_name is not None:
        CMR_KEYS.append("&options[readable_granule_name][pattern]=true")
        for gran in readable_granule_name:
            CMR_KEYS.append(f"&readable_granule_name[]={gran}")
    # full CMR query url
    cmr_query_url = "".join([posixpath.join(*CMR_HOST), *CMR_KEYS])
    logging.info(f"CMR request={cmr_query_url}")
    # output list of granule names and urls
    granule_urls = []
    cmr_search_after = None
    while True:
        req = urllib2.Request(cmr_query_url)
        # add CMR search after header
        if cmr_search_after:
            req.add_header("CMR-Search-After", cmr_search_after)
            logging.debug(f"CMR-Search-After: {cmr_search_after}")
        response = opener.open(req)
        # get search after index for next iteration
        headers = {k.lower(): v for k, v in dict(response.info()).items()}
        cmr_search_after = headers.get("cmr-search-after")
        # read the CMR search as JSON
        search_page = json.loads(response.read().decode("utf-8"))
        urls = cmr_filter_json(search_page, endpoint=endpoint)
        if not urls or cmr_search_after is None:
            break
        # extend lists
        granule_urls.extend(urls)
    # return the list of granule ids and urls
    return granule_urls


def fetch(path: pathlib.Path = get_cache_path(), **kwargs):
    """
    Query resources from the NASA Common Metadata Repository (CMR)
    and download them to a local path

    Parameters
    ----------
    path: str or pathlib.Path, default IceAdvect.utilities.get_cache_path()
        local path to download resources
    kwargs: dict
        keyword arguments for ``cmr``

    Returns
    -------
    granules: list
        local paths for queried resources
    """
    granules = []
    # for each url in the CMR query
    for url in cmr(**kwargs):
        # split the url into parts and get the granule name
        *_, file = url_split(url)
        # full path to output local file
        local = path.joinpath(file)
        # check if existing and download if not
        if not local.exists():
            from_earthdata(url, local=local)
        # append to list of granules
        granules.append(local)
    # return list of granules
    return granules
