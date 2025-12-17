<script lang="ts">
	import { toast } from 'svelte-sonner';
	import { createEventDispatcher, onMount, getContext } from 'svelte';

	const dispatch = createEventDispatcher();
	const i18n = getContext('i18n');

	import { user } from '$lib/stores';
	import { getGlobalConnections, type GlobalConnection } from '$lib/apis/connections';
	import { getUserApiKeys, updateUserApiKeys } from '$lib/apis/users';

	import Spinner from '$lib/components/common/Spinner.svelte';

	export let saveSettings: Function;

	let loading = true;
	let connections: GlobalConnection[] = [];
	let userApiKeys: Record<string, string> = {};

	onMount(async () => {
		await loadData();
	});

	const loadData = async () => {
		loading = true;
		try {
			// 加载全局连接列表
			connections = await getGlobalConnections(localStorage.token);
			// 加载用户的API密钥配置
			const keysData = await getUserApiKeys(localStorage.token);
			userApiKeys = keysData?.api_keys || {};
		} catch (error) {
			console.error('加载连接配置失败:', error);
			toast.error($i18n.t('Failed to load connections'));
		} finally {
			loading = false;
		}
	};

	const saveApiKeys = async () => {
		try {
			await updateUserApiKeys(localStorage.token, userApiKeys);
			toast.success($i18n.t('API keys saved'));
			dispatch('save');
		} catch (error) {
			console.error('保存API密钥失败:', error);
			toast.error($i18n.t('Failed to save API keys'));
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

	const getApiKeyLink = (url: string) => {
		try {
			const urlObj = new URL(url);
			return `${urlObj.protocol}//${urlObj.host}`;
		} catch {
			return null;
		}
	};
</script>

<form
	id="tab-connections"
	class="flex flex-col h-full justify-between text-sm"
	on:submit|preventDefault={saveApiKeys}
>
	<div class="overflow-y-scroll scrollbar-hidden h-full">
		{#if !loading}
			<div class="space-y-4">
				<div>
					<div class="font-medium mb-2">{$i18n.t('API 连接信息')}</div>

				</div>

				{#if connections.length === 0}
					<div class="text-center py-8 text-gray-500">
						<div class="text-lg mb-2">{$i18n.t('No connections available')}</div>
						<div class="text-sm">{$i18n.t('Please contact administrator to configure API connections.')}</div>
					</div>
				{:else}
					<div class="space-y-3">
						{#each connections as connection}
							<div class="border dark:border-gray-700 rounded-lg p-4">
								<div class="flex items-center gap-2 mb-3">
									<div class="font-medium">{connection.name}</div>
									<span class="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-800">
										{getConnectionTypeLabel(connection.type)}
									</span>
								</div>

								<div>
									<label class="block text-sm font-medium mb-1">{$i18n.t('API密钥')}</label>
									<input
										type="password"
										class="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
										placeholder={$i18n.t('请输入你的API密钥')}
										bind:value={userApiKeys[connection.id]}
									/>
								</div>

								{#if getApiKeyLink(connection.url)}
									<div class="mt-3 text-xs text-gray-500 dark:text-gray-400">
										{$i18n.t('密钥获取链接')}：<a
											href={getApiKeyLink(connection.url)}
											target="_blank"
											rel="noopener noreferrer"
											class="text-blue-500 hover:text-blue-600 hover:underline"
										>{getApiKeyLink(connection.url)}</a>
									</div>
								{/if}
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{:else}
			<div class="flex h-full justify-center">
				<div class="my-auto">
					<Spinner className="size-6" />
				</div>
			</div>
		{/if}
	</div>

	<div class="flex justify-end pt-3 text-sm font-medium">
		<button
			class="px-3.5 py-1.5 text-sm font-medium bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full"
			type="submit"
		>
			{$i18n.t('Save')}
		</button>
	</div>
</form>
