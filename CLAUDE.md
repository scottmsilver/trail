## Development Best Practices

- Use a venv
- Do test driven development
- Lets check in to git with small improvements each step of the way
- Always use a venv to run python code
- Always use a virtual env for node
- Use venv called trail_env
- Fix tests by fixing the logic or showing the test is invalid. Turning off a test to make it pass is against the rules.
- Tests should go into a tests directory. /tests
- Please run appropriate tests after each code change
- Put any debugging code into a debug directory. If it's temporary, use a tmp_debug directory

## Caching Strategy

- We don't do any caching for now. Fail requests when the data re not available. Require the caller to ask to load the data.