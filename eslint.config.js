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
		}
	},
	eslintConfigPrettier
)
