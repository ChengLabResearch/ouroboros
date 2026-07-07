const tseslint = require('@electron-toolkit/eslint-config-ts')
const eslintConfigPrettier = require('@electron-toolkit/eslint-config-prettier')
const eslintPluginReact = require('eslint-plugin-react')

module.exports = tseslint.config(
	{ ignores: ['node_modules', 'dist', 'out', 'python', 'plugins'] },
	tseslint.configs.recommended,
	eslintPluginReact.configs.flat.recommended,
	eslintPluginReact.configs.flat['jsx-runtime'],
	{
		settings: {
			react: { version: 'detect' }
		},
		rules: {
			'@typescript-eslint/no-unused-vars': [
				'error',
				{
					argsIgnorePattern: '^_',
					varsIgnorePattern: '^_',
					caughtErrorsIgnorePattern: '^_'
				}
			]
		}
	},
	// electron-toolkit/eslint-config-ts@3.1.0 intends to disable this rule for
	// .js/.mjs (see its eslint-typescript.js), but ships a legacy-eslint glob
	// (`*.mjs`) that only matches root-level under flat-config semantics, so
	// nested build scripts like `scripts/*.mjs` fall through. Restate the
	// intent with a `**/*.mjs` glob until upstream fixes the pattern.
	{
		files: ['**/*.mjs', '**/*.js'],
		rules: {
			'@typescript-eslint/explicit-function-return-type': 'off'
		}
	},
	{
		files: ['**/*.cjs'],
		rules: {
			'@typescript-eslint/no-require-imports': 'off'
		}
	},
	{
		files: ['**/*.tsx', '**/*.jsx'],
		rules: {
			'react/prop-types': 'off'
		}
	},
	eslintConfigPrettier
)
