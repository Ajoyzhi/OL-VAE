from mininet.topo import Topo

class MyTopo(Topo):
	def __init__(self):
		Topo.__init__(self)
		# 生成所需要的主机和交换机
		h1 = self.addHost( 'h1' )
		h2 = self.addHost( 'h2' )
		h3 = self.addHost( 'h3' )
		h4 = self.addHost( 'h4' )
		h5 = self.addHost( 'h5' )
		h6 = self.addHost( 'h6' )
		h7 = self.addHost( 'h7' )
		h8 = self.addHost( 'h8' )
		h9 = self.addHost( 'h9' )
		h10 = self.addHost( 'h10' )
		h11 = self.addHost( 'h11' )
		h12 = self.addHost( 'h12' )
		h13 = self.addHost( 'h13' )
		h14 = self.addHost( 'h14' )
		h15 = self.addHost( 'h15' )

		sw1= self.addSwitch( 'sw1' )
		sw2 = self.addSwitch( 'sw2' )
		sw3= self.addSwitch( 'sw3' )
		sw4= self.addSwitch( 'sw4' )
		sw5= self.addSwitch( 'sw5' )
 
		# 添加连线，交换机和交换机之间，交换机和主机之间
		self.addLink( sw1, sw2)
		self.addLink( sw1, sw3)
		self.addLink( sw3, sw4)
		self.addLink( sw3, sw5)

		self.addLink( sw1, h1)
		self.addLink( sw1, h2)
		self.addLink( sw1, h3)
		self.addLink( sw2, h4)
		self.addLink( sw2, h5)
		self.addLink( sw2, h6)
		self.addLink( sw3, h7)
		self.addLink( sw3, h8)
		self.addLink( sw3, h9)
		self.addLink( sw4, h10)
		self.addLink( sw4, h11)
		self.addLink( sw4, h12)
		self.addLink( sw5, h13)
		self.addLink( sw5, h14)
		self.addLink( sw5, h15)
 
#实例化类
topos = { 'mytopo': ( lambda: MyTopo() ) }

