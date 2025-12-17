<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';
	import {
		getGlobalConnections,
		createGlobalConnection,
		updateGlobalConnection,
		deleteGlobalConnection,
		type GlobalConnection,
		type GlobalConnectionForm
	} from '$lib/apis/connections';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	let loading = false;
	let connections: GlobalConnection[] = [];

	// 编辑模态框状态
	let showEditModal = false;
	let editingConnection: GlobalConnection | null = null;
	let connectionForm: GlobalConnectionForm & { api_key?: string } = {
		name: '',
		type: 'openai',
		url: '',
		api_key: '',
		auth_type: 'bearer'
	};

	onMount(async () => {
		await loadConnections();
	});

	const loadConnections = async () => {
		loading = true;
		try {
			connections = await getGlobalConnections(localStorage.token);
		} catch (error) {
			console.error('加载全局连接失败:', error);
			toast.error($i18n.t('Failed to load global connections'));
		} finally {
			loading = false;
		}
	};

	const openAddModal = () => {
		editingConnection = null;
		connectionForm = {
			name: '',
			type: 'openai',
			url: '',
			api_key: '',
			auth_type: 'bearer'
		};
		showEditModal = true;
	};

	const openEditModal = (connection: GlobalConnection) => {
		editingConnection = connection;
		connectionForm = {
			name: connection.name,
			type: connection.type,
			url: connection.url,
			api_key: (connection as any).api_key || '',
			auth_type: connection.auth_type,
			config: connection.config
		};
		showEditModal = true;
	};

	const closeModal = () => {
		showEditModal = false;
		editingConnection = null;
	};

	const saveConnection = async () => {
		if (!connectionForm.name || !connectionForm.url) {
			toast.error($i18n.t('Please fill in connection name and URL'));
			return;
		}

		loading = true;
		try {
			if (editingConnection) {
				// 更新连接
				await updateGlobalConnection(localStorage.token, editingConnection.id, connectionForm);
				toast.success($i18n.t('Connection updated'));
			} else {
				// 创建新连接
				await createGlobalConnection(localStorage.token, connectionForm);
				toast.success($i18n.t('Connection created'));
			}

			closeModal();
			await loadConnections();
			// 通知父组件刷新模型列表
			dispatch('updated');
		} catch (error: any) {
			console.error('保存连接失败:', error);
			toast.error(error?.detail || $i18n.t('Failed to save connection'));
		} finally {
			loading = false;
		}
	};

	const deleteConnection = async (connection: GlobalConnection) => {
		if (!confirm($i18n.t('Are you sure you want to delete connection "{{name}}"?', { name: connection.name }))) {
			return;
		}

		loading = true;
		try {
			await deleteGlobalConnection(localStorage.token, connection.id);
			toast.success($i18n.t('Connection deleted'));
			await loadConnections();
			// 通知父组件刷新模型列表
			dispatch('updated');
		} catch (error) {
			console.error('删除连接失败:', error);
			toast.error($i18n.t('Failed to delete connection'));
		} finally {
			loading = false;
		}
	};

	const getConnectionTypeLabel = (type: string) => {
		const types: Record<string, string> = {
			openai: 'OpenAI',
			anthropic: 'Anthropic/Claude',
			ollama: 'Ollama',
			custom: $i18n.t('Custom')
		};
		return types[type] || type;
	};
</script>

<div class="flex flex-col gap-4">
	<div class="flex justify-between items-center">
		<div>
			<div class="text-lg font-medium">{$i18n.t('Global API Connections')}</div>
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t('Configure API connections with URLs and API keys for all users')}
			</div>
		</div>
		<button
			class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
			on:click={openAddModal}
			disabled={loading}
		>
			+ {$i18n.t('Add Connection')}
		</button>
	</div>

	{#if loading && connections.length === 0}
		<div class="flex justify-center items-center py-8">
			<div class="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900 dark:border-gray-100"></div>
		</div>
	{:else if connections.length === 0}
		<div class="text-center py-8 text-gray-500">
			<div class="text-lg mb-2">{$i18n.t('No global connections')}</div>
			<div class="text-sm">{$i18n.t('Click "Add Connection" to create the first API connection')}</div>
		</div>
	{:else}
		<div class="space-y-3">
			{#each connections as connection}
				<div class="border dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50">
					<div class="flex items-start justify-between">
						<div class="flex-1">
							<div class="flex items-center gap-2 mb-2">
								<div class="font-medium">{connection.name}</div>
								<span class="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800">
									{getConnectionTypeLabel(connection.type)}
								</span>
							</div>
							<div class="text-sm text-gray-600 dark:text-gray-400 space-y-1">
								<div>
									<span class="font-medium">URL:</span>
									<span class="ml-2 font-mono text-xs">{connection.url}</span>
								</div>
								<div>
									<span class="font-medium">{$i18n.t('API Key')}:</span>
									<span class="ml-2 font-mono text-xs">
										{(connection as any).api_key ? '••••••••' : $i18n.t('Not configured')}
									</span>
								</div>
							</div>
						</div>

						<div class="flex gap-2">
							<button
								class="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
								on:click={() => openEditModal(connection)}
								disabled={loading}
							>
								{$i18n.t('Edit')}
							</button>
							<button
								class="px-3 py-1 text-sm border border-red-500 text-red-500 rounded hover:bg-red-50 dark:hover:bg-red-900/20"
								on:click={() => deleteConnection(connection)}
								disabled={loading}
							>
								{$i18n.t('Delete')}
							</button>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<!-- 编辑/添加连接模态框 -->
{#if showEditModal}
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50" on:click={closeModal}>
		<div class="bg-white dark:bg-gray-900 rounded-lg shadow-xl max-w-2xl w-full mx-4 p-6" on:click|stopPropagation>
			<div class="flex justify-between items-center mb-4">
				<h2 class="text-xl font-semibold">
					{editingConnection ? $i18n.t('Edit Connection') : $i18n.t('Add Connection')}
				</h2>
				<button
					class="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
					on:click={closeModal}
				>
					✕
				</button>
			</div>

			<div class="space-y-4">
				<div>
					<label class="block text-sm font-medium mb-1">{$i18n.t('Connection Name')}</label>
					<input
						type="text"
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder={$i18n.t('e.g., OpenAI Official')}
						bind:value={connectionForm.name}
					/>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">{$i18n.t('Connection Type')}</label>
					<select
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
						bind:value={connectionForm.type}
					>
						<option value="openai">OpenAI</option>
						<option value="anthropic">Anthropic/Claude</option>
						<option value="ollama">Ollama</option>
						<option value="custom">{$i18n.t('Custom')}</option>
					</select>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">API URL</label>
					<input
						type="url"
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder={connectionForm.type === 'anthropic' ? 'https://api.anthropic.com/v1/messages' : 'https://api.openai.com/v1'}
						bind:value={connectionForm.url}
					/>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">{$i18n.t('API Key')}</label>
					<input
						type="password"
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
						placeholder={$i18n.t('Enter API key')}
						bind:value={connectionForm.api_key}
					/>
					<div class="text-xs text-gray-500 mt-1">
						{$i18n.t('API key will be used for all users. Leave empty if not required.')}
					</div>
				</div>

				<div>
					<label class="block text-sm font-medium mb-1">{$i18n.t('Auth Type')}</label>
					<select
						class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
						bind:value={connectionForm.auth_type}
					>
						<option value="bearer">Bearer Token</option>
						<option value="none">{$i18n.t('No Auth')}</option>
						<option value="session">{$i18n.t('Session Auth')}</option>
					</select>
				</div>
			</div>

			<div class="flex justify-end gap-2 mt-6">
				<button
					class="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
					on:click={closeModal}
					disabled={loading}
				>
					{$i18n.t('Cancel')}
				</button>
				<button
					class="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
					on:click={saveConnection}
					disabled={loading || !connectionForm.name || !connectionForm.url}
				>
					{editingConnection ? $i18n.t('Update') : $i18n.t('Create')}
				</button>
			</div>
		</div>
	</div>
{/if}
