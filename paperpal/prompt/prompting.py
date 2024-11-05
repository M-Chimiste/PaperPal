import jinja2
from jinja2 import meta
import inspect
from functools import wraps


def prompt(func):
    """Decorator that wraps a function to enable Jinja2 templating of its docstring."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function that templates the docstring of the wrapped function."""
        # get the function's docstring.
        docstring = func.__doc__
        # map args and kwargs onto func's signature.
        signature = inspect.signature(func)
        bound_arguments = signature.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        kwargs = bound_arguments.arguments

        # create a Jinja2 environment
        env = jinja2.Environment()

        # parse the docstring
        parsed_content = env.parse(docstring)

        # get all variables in the docstring
        variables = meta.find_undeclared_variables(parsed_content)

        # check if all variables are in kwargs
        for var in variables:
            if var not in kwargs:
                raise ValueError(f"Variable '{var}' was not passed into the function")

        # interpolate docstring with args and kwargs
        template = env.from_string(docstring)
        return template.render(**kwargs)
    return wrapper